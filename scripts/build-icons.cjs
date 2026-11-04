const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');

const root = path.resolve(__dirname, '..');

// Electron supplies the same PNG decoder and high-quality resize used by the app.
if (!process.versions.electron) {
  const env = { ...process.env };
  delete env.ELECTRON_RUN_AS_NODE;
  const result = spawnSync(require('electron'), [__filename], { cwd: root, env, windowsHide: true, stdio: 'inherit' });
  if (result.error) throw result.error;
  process.exit(result.status ?? 1);
}

const { app, nativeImage } = require('electron');
const { Data } = require('resedit');

function master(name) {
  const image = nativeImage.createFromPath(path.join(root, 'assets', 'branding', name));
  const { width, height } = image.getSize();
  if (image.isEmpty() || width !== height || width < 256) throw new Error(`${name}: expected a square PNG of at least 256px`);
  return image;
}

function writeIco(name, image, sizes) {
  const ico = new Data.IconFile();
  ico.icons = sizes.map(size => ({
    data: Data.RawIconItem.from(image.resize({ width: size, height: size, quality: 'best' }).toPNG(), size, size, 32),
  }));
  fs.writeFileSync(path.join(root, 'electron', name), Buffer.from(ico.generate()));
}

app.whenReady().then(() => {
  const application = master('app-master.png');
  const tray = master('tray-master.png');
  fs.writeFileSync(path.join(root, 'electron', 'icon.png'), application.resize({ width: 256, height: 256, quality: 'best' }).toPNG());
  writeIco('icon.ico', application, [16, 20, 24, 32, 40, 48, 64, 128, 256]);
  writeIco('tray.ico', tray, [16, 20, 24, 32, 40, 48]);
  console.log('Generated electron/icon.png, electron/icon.ico and electron/tray.ico');
  app.quit();
}).catch(error => { console.error(error); app.exit(1); });
