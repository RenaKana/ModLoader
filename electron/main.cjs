const { app, BrowserWindow, Menu, Tray, nativeImage, ipcMain, dialog, shell, clipboard, screen } = require('electron');
const { spawn } = require('node:child_process');
const path = require('node:path');
const fs = require('node:fs');
const crypto = require('node:crypto');
const readline = require('node:readline');
const { pathToFileURL } = require('node:url');
const { createMiniMonitor } = require('./mini-monitor.cjs');

const projectRoot = path.resolve(__dirname, '..');
const dataRoot = process.env.MODLOADER_DATA_DIR || (app.isPackaged ? path.join(path.dirname(app.getPath('exe')), 'data') : path.join(projectRoot,'data'));
app.setPath('userData',path.join(dataRoot,'desktop'));
app.setPath('sessionData',path.join(dataRoot,'desktop','session'));
let core, window, tray, connection, quitting = false;
let lifecycleRequest = null;
let pendingQuitIntent = false;
let rendererDocumentLoaded = false;
let rendererFlushReady = false;
const appearancePath=path.join(dataRoot,'desktop','appearance.json');
let appearance = {};
let appearanceSaveTimer = null;
const MIN_WINDOW_WIDTH = 940;
const MIN_WINDOW_HEIGHT = 640;
const BEFORE_CLOSE_REQUEST = 'modloader:before-close-request';
const BEFORE_CLOSE_RESULT = 'modloader:before-close-result';
const BEFORE_CLOSE_READY = 'modloader:before-close-ready';
const CLOSE_CANCELLED = 'modloader:close-cancelled';
const BEFORE_CLOSE_TIMEOUT_MS = 8000;
function readAppearance() {
  try {
    const saved = JSON.parse(fs.readFileSync(appearancePath, 'utf8'));
    return saved && typeof saved === 'object' && !Array.isArray(saved) ? saved : {};
  } catch {
    return {};
  }
}

function restoreWindowBounds(bounds) {
  const valid = bounds && ['x', 'y', 'width', 'height'].every(key => Number.isSafeInteger(bounds[key]))
    && bounds.width > 0 && bounds.height > 0;
  const area = (valid ? screen.getDisplayMatching(bounds) : screen.getPrimaryDisplay()).workArea;
  // Keep visible user-selected bounds, including windows spanning displays.
  if (valid && bounds.x < area.x + area.width && bounds.x + bounds.width > area.x
      && bounds.y < area.y + area.height && bounds.y + bounds.height > area.y) {
    return {
      x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height,
      minWidth: Math.min(MIN_WINDOW_WIDTH, area.width),
      minHeight: Math.min(MIN_WINDOW_HEIGHT, area.height),
    };
  }
  const width = Math.min(area.width, Math.max(MIN_WINDOW_WIDTH, valid ? bounds.width : 1440));
  const height = Math.min(area.height, Math.max(MIN_WINDOW_HEIGHT, valid ? bounds.height : 960));
  return {
    ...(valid ? {
      x: Math.max(area.x, Math.min(bounds.x, area.x + area.width - width)),
      y: Math.max(area.y, Math.min(bounds.y, area.y + area.height - height)),
    } : {}),
    width,
    height,
    minWidth: Math.min(MIN_WINDOW_WIDTH, area.width),
    minHeight: Math.min(MIN_WINDOW_HEIGHT, area.height),
  };
}

function rememberAppearance() {
  if (!window || window.isDestroyed()) return;
  // Minimized bounds and flags are transient; reopen the last visible state.
  if (!window.isMinimized()) {
    // On Windows, fullscreen normal bounds describe the pre-fullscreen maximized rectangle.
    if (!window.isFullScreen()) {
      appearance.bounds = window.getNormalBounds();
      appearance.maximized = window.isMaximized();
    }
    appearance.fullScreen = window.isFullScreen();
  }
  appearance.zoomFactor = window.webContents.getZoomFactor();
}

function saveAppearance() {
  clearTimeout(appearanceSaveTimer);
  appearanceSaveTimer = null;
  if (!window || window.isDestroyed()) return;
  rememberAppearance();
  try {
    fs.mkdirSync(path.dirname(appearancePath), { recursive: true });
    fs.writeFileSync(appearancePath, JSON.stringify(appearance));
  } catch (error) {
    console.error('窗口状态保存失败：', error);
  }
}

function scheduleAppearanceSave() {
  rememberAppearance();
  clearTimeout(appearanceSaveTimer);
  appearanceSaveTimer = setTimeout(saveAppearance, 200);
}
const token = crypto.randomBytes(32).toString('base64url');
if (!app.requestSingleInstanceLock()) app.quit();
app.on('second-instance',()=>{window?.show();window?.focus();});

async function startCore() {
  const bundled = path.join(process.resourcesPath,'core','modloader-core.exe');
  const executable = app.isPackaged ? bundled : path.join(projectRoot,'.venv','Scripts','python.exe');
  if (!fs.existsSync(executable)) throw new Error('未找到核心运行环境。请先安装项目 .venv 或重新生成桌面包。');
  const args = [...(app.isPackaged ? [] : ['-m','modloader']), '--data-dir',dataRoot,'--parent-pipe'];
  const env = {...process.env, MODLOADER_MANAGEMENT_TOKEN:token, PYTHONPATH:path.join(projectRoot,'core'), PYTHONUNBUFFERED:'1', PYTHONUTF8:'1'};
  core=spawn(executable,args,{cwd:app.isPackaged?path.dirname(bundled):projectRoot,env,windowsHide:true,stdio:['pipe','pipe','pipe']});
  let errors='';
  core.stderr.on('data',chunk=>{errors=(errors+chunk.toString()).slice(-12000);});
  core.on('exit',()=>{
    if (!quitting && window) dialog.showErrorBox('核心服务已停止','请重新启动 ModLoader。正在运行的模型已由进程管理器回收。');
  });
  return new Promise((resolve,reject)=>{
    const timer=setTimeout(()=>reject(new Error('核心启动超时\n'+errors)),45000);
    const reader=readline.createInterface({input:core.stdout});
    reader.on('line',line=>{
      try { const result=JSON.parse(line); if(result.event==='ready') {clearTimeout(timer);connection={baseUrl:`http://127.0.0.1:${result.port}`,token};resolve(connection);} } catch {}
    });
    core.once('error',error=>{clearTimeout(timer);reject(error);});
    core.once('exit',code=>{clearTimeout(timer);reject(new Error(`核心启动失败 (${code})\n${errors}`));});
  });
}

function windowIcon() {
  return nativeImage.createFromPath(path.join(__dirname,'icon.png'));
}

function isTrustedMainFrame(event) {
  return !!window && !window.isDestroyed() && !window.webContents.isDestroyed()
    && event.sender === window.webContents && event.senderFrame === window.webContents.mainFrame;
}

function requireMainFrame(event) {
  if (!isTrustedMainFrame(event)) throw new Error('只允许从主界面执行此操作。');
}

function requestRendererFlush() {
  if (!window || window.isDestroyed() || window.webContents.isDestroyed() || !rendererDocumentLoaded) {
    return Promise.resolve({ ok: false, reason: 'unavailable' });
  }
  if (!rendererFlushReady) return Promise.resolve({ ok: false, reason: 'handler-unavailable' });

  const requestId = crypto.randomUUID();
  return new Promise(resolve => {
    let settled = false;
    const finish = result => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      ipcMain.removeListener(BEFORE_CLOSE_RESULT, onResult);
      resolve(result);
    };
    const onResult = (event, responseId, success) => {
      if (!isTrustedMainFrame(event) || responseId !== requestId) return;
      finish(success === true ? { ok: true } : { ok: false, reason: 'failed' });
    };
    const timer = setTimeout(() => finish({ ok: false, reason: 'timeout' }), BEFORE_CLOSE_TIMEOUT_MS);
    ipcMain.on(BEFORE_CLOSE_RESULT, onResult);
    try {
      window.webContents.send(BEFORE_CLOSE_REQUEST, requestId);
    } catch {
      finish({ ok: false, reason: 'failed' });
    }
  });
}

async function confirmProceedAfterFlushFailure(action, reason) {
  const isQuit = action === 'quit';
  const timedOut = reason === 'timeout';
  const options = {
    type: 'warning',
    title: isQuit ? '退出 ModLoader' : '重新加载界面',
    message: '更改尚未保存',
    detail: `${timedOut ? '等待保存超时（8 秒）。' : '保存失败。'}仍然${isQuit ? '退出' : '重新加载'}可能丢失未保存的设置或聊天记录。`,
    buttons: isQuit ? ['取消退出', '仍然退出'] : ['取消重新加载', '仍然重新加载'],
    defaultId: 0,
    cancelId: 0,
    noLink: true,
  };
  const owner = window && !window.isDestroyed() ? window : null;
  const result = owner
    ? await dialog.showMessageBox(owner, options)
    : await dialog.showMessageBox(options);
  return result.response === 1;
}

function notifyCloseCancelled() {
  if (!window || window.isDestroyed() || window.webContents.isDestroyed()) return;
  window.webContents.send(CLOSE_CANCELLED);
}

async function stopCoreAndQuit() {
  if (quitting) return;
  quitting = true;
  saveAppearance();
  tray?.destroy();
  if (core && core.exitCode === null) {
    core.stdin.end();
    await new Promise(resolve=>{const timer=setTimeout(()=>{core.kill();resolve();},25000);core.once('exit',()=>{clearTimeout(timer);resolve();});});
  }
  app.quit();
}

function finishLifecycleRequest() {
  lifecycleRequest = null;
  if (pendingQuitIntent && !quitting) {
    pendingQuitIntent = false;
    void quitSafely();
  }
}

async function quitSafely({ skipFlush = false } = {}) {
  if (quitting) return;
  if (lifecycleRequest) {
    if (lifecycleRequest === 'reload') pendingQuitIntent = true;
    return;
  }
  lifecycleRequest = 'quit';
  try {
    if (!skipFlush) {
      const flush = await requestRendererFlush();
      if (flush.reason && flush.reason !== 'unavailable') {
        const proceed = await confirmProceedAfterFlushFailure('quit', flush.reason);
        if (!proceed) {
          notifyCloseCancelled();
          return;
        }
      }
    }
    await stopCoreAndQuit();
  } finally {
    finishLifecycleRequest();
  }
}

async function reloadSafely() {
  if (quitting || lifecycleRequest || !window || window.isDestroyed()) return;
  lifecycleRequest = 'reload';
  try {
    const flush = await requestRendererFlush();
    if (flush.reason && flush.reason !== 'unavailable') {
      const proceed = await confirmProceedAfterFlushFailure('reload', flush.reason);
      if (!proceed) {
        notifyCloseCancelled();
        return;
      }
    }
    if (!window || window.isDestroyed() || window.webContents.isDestroyed()) return;
    rendererDocumentLoaded = false;
    rendererFlushReady = false;
    window.webContents.reload();
  } finally {
    finishLifecycleRequest();
  }
}

app.whenReady().then(async()=>{
  try {
    await startCore();
    ipcMain.handle('connection',event=>{requireMainFrame(event);return connection;});
    ipcMain.handle('pick-path',async(event,kind)=>{requireMainFrame(event);const result=await dialog.showOpenDialog(window,{properties:[kind==='directory'?'openDirectory':'openFile']});return result.canceled?null:result.filePaths[0];});
    ipcMain.handle('copy',(event,text)=>{requireMainFrame(event);return clipboard.writeText(String(text));});
    ipcMain.handle('open-external',(event,url)=>{requireMainFrame(event);const parsed=new URL(url);if(!['http:','https:'].includes(parsed.protocol))throw new Error('只允许打开网页');return shell.openExternal(parsed.href);});
    ipcMain.handle('open-path',(event,target)=>{requireMainFrame(event);return shell.showItemInFolder(path.resolve(target));});
    ipcMain.handle('local-resource-action',async(event,id,action)=>{
      requireMainFrame(event);
      if(typeof id!=='string' || !/^[A-Za-z0-9_-]{43}$/.test(id) || !['open','reveal'].includes(action))throw new Error('附件操作参数无效。');
      // The core resolves the persisted reference against the current grant and
      // pins its path while handing it to Windows; renderer paths are never used.
      const response=await fetch(`${connection.baseUrl.replace(/\/$/,'')}/api/chat-tools/resources/${id}/action`,{
        method:'POST',headers:{Authorization:`Bearer ${connection.token}`,'Content-Type':'application/json'},body:JSON.stringify({action}),signal:AbortSignal.timeout(30000)
      });
      const result=await response.json();
      if(!response.ok)throw new Error(typeof result.error==='string'?result.error:typeof result.detail==='string'?result.detail:'附件操作失败。');
      return result;
    });
    ipcMain.handle('export-html',async(event,source,action)=>{
      if(event.sender!==window?.webContents || event.senderFrame!==window.webContents.mainFrame)throw new Error('只允许从主界面导出 HTML。');
      if(typeof source!=='string' || !['open','save'].includes(action))throw new Error('HTML 导出参数无效。');
      let target;
      if(action==='save'){
        const result=await dialog.showSaveDialog(window,{title:'保存 HTML',defaultPath:'preview.html',filters:[{name:'HTML 网页',extensions:['html']} ]});
        if(result.canceled || !result.filePath)return {canceled:true};
        target=result.filePath;
      }else{
        const directory=await fs.promises.mkdtemp(path.join(app.getPath('temp'),'modloader-html-'));
        target=path.join(directory,'preview.html');
      }
      // UTF-8 BOM also handles fragments without a charset declaration. Preserve the original markup and scripts.
      await fs.promises.writeFile(target,'\ufeff'+source,'utf8');
      if(action==='open')await shell.openExternal(pathToFileURL(target).href);
      return {canceled:false};
    });
    ipcMain.handle('export-markdown',async(event,source,title)=>{
      if(!isTrustedMainFrame(event))throw new Error('只允许从主界面导出对话。');
      if(typeof source!=='string' || typeof title!=='string')throw new Error('对话导出参数无效。');
      let filename=title.replace(/[<>:"/\\|?*\x00-\x1f]/g,'_').replace(/[. ]+$/g,'').trim() || '对话';
      if(/^(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)/i.test(filename))filename='_'+filename;
      const result=await dialog.showSaveDialog(window,{title:'导出对话为 Markdown',defaultPath:filename+'.md',filters:[{name:'Markdown 文档',extensions:['md']}]});
      if(result.canceled || !result.filePath)return {canceled:true};
      await fs.promises.writeFile(result.filePath,source,'utf8');
      return {canceled:false};
    });
    ipcMain.handle('save-image',async(event,source,name)=>{
      requireMainFrame(event);
      if(typeof source!=='string' || typeof name!=='string')throw new Error('图片保存参数无效。');
      const embedded=/^data:(image\/[a-z\d.+-]+);base64,([a-z\d+/=\s]+)$/i.exec(source);
      let bytes, mime, url;
      if(embedded){
        mime=embedded[1].toLowerCase();
        bytes=Buffer.from(embedded[2].replace(/\s/g,''),'base64');
      }else{
        try{url=new URL(source);}catch{throw new Error('图片地址无效。');}
        if(!['http:','https:'].includes(url.protocol))throw new Error('只允许保存内嵌图片或 HTTP(S) 原图。');
        try{
          // Use the image's Chromium session, without renderer CORS or canvas conversion.
          const response=await window.webContents.session.fetch(url.href,{signal:AbortSignal.timeout(60000),credentials:'include',referrerPolicy:'no-referrer'});
          if(!response.ok)throw new Error(`图片下载失败（HTTP ${response.status}）。`);
          mime=(response.headers.get('content-type') || '').split(';')[0].trim().toLowerCase();
          bytes=Buffer.from(await response.arrayBuffer());
        }catch(error){
          if(error.name==='TimeoutError' || error.name==='AbortError')throw new Error('图片下载超时，请重试。');
          throw error;
        }
      }
      if(!bytes.length)throw new Error('图片内容为空。');
      const mimeExtensions={'image/png':'png','image/jpeg':'jpg','image/gif':'gif','image/webp':'webp','image/avif':'avif','image/bmp':'bmp','image/svg+xml':'svg','image/x-icon':'ico','image/vnd.microsoft.icon':'ico','image/tiff':'tiff','image/apng':'apng'};
      const knownExtensions=/^(png|jpe?g|gif|webp|avif|bmp|svg|ico|tiff?|apng)$/i;
      const urlExtension=url && path.extname(url.pathname).slice(1);
      const extension=mimeExtensions[mime] || (knownExtensions.test(urlExtension || '')?urlExtension.toLowerCase():'img');
      let filename=name.replace(/[<>:"/\\|?*\x00-\x1f]/g,'_').replace(/[. ]+$/g,'').trim() || '图片';
      if(/^(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)/i.test(filename))filename='_'+filename;
      const currentExtension=path.extname(filename).slice(1).toLowerCase();
      if(currentExtension!==extension && !(extension==='jpg' && currentExtension==='jpeg')){
        if(knownExtensions.test(currentExtension))filename=filename.slice(0,-currentExtension.length-1);
        filename+='.'+extension;
      }
      const result=await dialog.showSaveDialog(window,{title:'保存图片',defaultPath:filename,filters:[{name:'原始图片',extensions:[extension]},{name:'所有文件',extensions:['*']}]});
      if(result.canceled || !result.filePath)return {canceled:true};
      const temporary=path.join(path.dirname(result.filePath),`.modloader-image-${crypto.randomUUID()}.tmp`);
      try{
        await fs.promises.writeFile(temporary,bytes,{flag:'wx'});
        await fs.promises.rename(temporary,result.filePath);
      }finally{
        await fs.promises.unlink(temporary).catch(error=>{if(error.code!=='ENOENT')throw error;});
      }
      return {canceled:false};
    });
    // Keep the one-second monitoring polls active when occluded or minimized.
    appearance = readAppearance();
    const bounds = restoreWindowBounds(appearance.bounds);
    const zoomFactor = Number.isFinite(appearance.zoomFactor) && appearance.zoomFactor > 0 ? appearance.zoomFactor : 1;
    window=new BrowserWindow({...bounds,title:'ModLoader',backgroundColor:'#f7f9fb',icon:windowIcon(),show:false,webPreferences:{zoomFactor,preload:path.join(__dirname,'preload.cjs'),contextIsolation:true,nodeIntegration:false,sandbox:true,backgroundThrottling:false}});
    // Windows caps constructor dimensions to one display; setBounds preserves spanning windows.
    if (bounds.x !== undefined) window.setBounds({ x: bounds.x, y: bounds.y, width: bounds.width, height: bounds.height });
    appearance.bounds = window.getNormalBounds();
    ipcMain.on(BEFORE_CLOSE_READY, (event, ready) => {
      if (!isTrustedMainFrame(event) || typeof ready !== 'boolean') return;
      rendererFlushReady = ready;
    });
    window.webContents.on('did-start-navigation', details => {
      if (!details.isMainFrame || details.isSameDocument) return;
      rendererDocumentLoaded = false;
      rendererFlushReady = false;
    });
    window.webContents.on('did-finish-load', () => {
      rendererDocumentLoaded = true;
    });
    window.webContents.on('zoom-changed',(_,direction)=>{
      window.webContents.setZoomLevel(window.webContents.getZoomLevel()+(direction==='in'?1:-1));
      saveAppearance();
    });
    window.webContents.setWindowOpenHandler(({url})=>{if(/^https?:\/\//.test(url))shell.openExternal(url);return {action:'deny'};});
    window.webContents.on('will-navigate',(event,url)=>{if(!url.startsWith('file://')&&!url.startsWith('http://127.0.0.1:5173/'))event.preventDefault();});
    Menu.setApplicationMenu(Menu.buildFromTemplate([
      {label:'文件',submenu:[{label:'退出 ModLoader',click:quitSafely}]},
      {label:'视图',submenu:[{label:'重新加载界面',accelerator:'CommandOrControl+R',click:reloadSafely},{role:'resetZoom',label:'实际大小'},{role:'zoomIn',label:'放大'},{role:'zoomOut',label:'缩小'},{type:'separator'},{role:'togglefullscreen',label:'切换全屏',accelerator:'F11'},{role:'toggleDevTools',label:'开发者工具'}]}
    ]));
    const miniMonitor = createMiniMonitor({mainWindow:window,dataRoot,projectRoot,icon:windowIcon(),devUrl:process.env.MODLOADER_DEV_URL,isTrustedMainFrame});
    window.on('close',event=>{
      saveAppearance();
      if(quitting)return;
      event.preventDefault();
      fetch(connection.baseUrl+'/api/state',{headers:{Authorization:'Bearer '+token}}).then(r=>r.json()).then(state=>{if(state.settings.close_to_tray)window.hide();else quitSafely();}).catch(()=>quitSafely());
    });
    tray=new Tray(path.join(__dirname,'tray.ico'));
    tray.setToolTip('ModLoader · 本地模型工作台');
    tray.setContextMenu(Menu.buildFromTemplate([{label:'打开 ModLoader',click:()=>window.show()},{label:'打开监控小窗',click:()=>miniMonitor.open().catch(error=>dialog.showErrorBox('监控小窗启动失败',String(error.message||error)))},{type:'separator'},{label:'退出并停止模型',click:quitSafely}]));
    tray.on('double-click',()=>window.show());
    if (process.env.MODLOADER_DEV_URL) await window.loadURL(process.env.MODLOADER_DEV_URL);
    else await window.loadFile(path.join(projectRoot,'dist','index.html'));
    if (appearance.maximized === true) window.maximize();
    if (appearance.fullScreen === true) window.setFullScreen(true);
    window.show();
    for (const event of ['move', 'resize', 'maximize', 'unmaximize', 'enter-full-screen', 'leave-full-screen', 'restore']) {
      window.on(event, scheduleAppearanceSave);
    }
    for (const event of ['show', 'hide', 'minimize']) window.on(event, saveAppearance);
    window.on('hide', () => { if (!window.webContents.isDestroyed()) window.webContents.send('modloader:window-hidden'); });
    saveAppearance();
  } catch(error) {
    dialog.showErrorBox('ModLoader 启动失败',String(error.message||error));
    await quitSafely({skipFlush:true});
  }
});
app.on('before-quit',event=>{
  if (quitting) return;
  event.preventDefault();
  if (lifecycleRequest === 'reload') {
    pendingQuitIntent = true;
    return;
  }
  void quitSafely();
});
app.on('window-all-closed',()=>{});
