import json
import sqlite3
import threading
from pathlib import Path

UNFINISHED_JOBS = ('queued', 'running', 'pausing', 'paused', 'cancelling', 'awaiting_confirmation')
TERMINAL_JOBS = ('completed', 'complete', 'failed', 'error', 'cancelled', 'canceled', 'interrupted')


class Store:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.RLock()
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("CREATE TABLE IF NOT EXISTS records (kind TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY(kind,id))")
        self.connection.commit()

    def all(self, kind):
        with self.lock:
            return [json.loads(row[0]) for row in self.connection.execute("SELECT body FROM records WHERE kind=? ORDER BY rowid", (kind,))]

    def get(self, kind, identifier):
        with self.lock:
            row = self.connection.execute("SELECT body FROM records WHERE kind=? AND id=?", (kind, identifier)).fetchone()
            return json.loads(row[0]) if row else None

    def put(self, kind, record):
        with self.lock, self.connection:
            self.connection.execute("INSERT INTO records(kind,id,body) VALUES(?,?,?) ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body", (kind, record['id'], json.dumps(record, ensure_ascii=False)))
        return record

    def remove(self, kind, identifier):
        with self.lock, self.connection:
            self.connection.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, identifier))

    def jobs(self, kind, states=None):
        query = "SELECT body FROM records WHERE kind='jobs' AND json_extract(body,'$.kind')=?"
        args = [kind]
        if states is not None:
            query += " AND json_extract(body,'$.state') IN (" + ','.join('?' for _ in states) + ")"
            args.extend(states)
        with self.lock:
            return [json.loads(row[0]) for row in self.connection.execute(query + ' ORDER BY rowid', args)]

    def job_history(self, kind, states, page, page_size=10):
        where = "kind='jobs' AND json_extract(body,'$.kind')=? AND json_extract(body,'$.state') IN (" + ','.join('?' for _ in states) + ')'
        args = [kind, *states]
        with self.lock:
            total = self.connection.execute('SELECT COUNT(*) FROM records WHERE ' + where, args).fetchone()[0]
            items = [json.loads(row[0]) for row in self.connection.execute(
                'SELECT body FROM records WHERE ' + where + ' ORDER BY rowid DESC LIMIT ? OFFSET ?',
                [*args, page_size, (page - 1) * page_size])]
        return {'items': items, 'total': total, 'page': page, 'page_size': page_size}

    def apply_runtime_switch(self, expected, replacements, remove_runtime=None):
        """Compare and save a selected group together; never partially migrate profiles."""
        with self.lock, self.connection:
            for original in expected:
                current = self.get('profiles', original['id'])
                if current != original:
                    raise ValueError('加载配置已变化，请重新检查后切换')
            for record in replacements:
                self.connection.execute("UPDATE records SET body=? WHERE kind='profiles' AND id=?",
                                        (json.dumps(record, ensure_ascii=False), record['id']))
            if remove_runtime:
                expected_ids = {record['id'] for record in expected}
                if any(p.get('runtime_id') == remove_runtime and p['id'] not in expected_ids for p in self.all('profiles')):
                    raise ValueError('环境新增了关联配置，请重新检查')
                self.connection.execute("DELETE FROM records WHERE kind='runtimes' AND id=?", (remove_runtime,))

    def close(self):
        self.connection.close()

    def apply_model_storage_migration(self, expected_model, expected_profiles, expected_runtimes,
                                      new_model, replacements, expected_preparations=None,
                                      new_preparations=None):
        """Register the copied model and switch its selected profiles atomically."""
        with self.lock, self.connection:
            if self.get('models', expected_model['id']) != expected_model:
                raise ValueError('模型登记已变化，请重新检查迁移清单')
            current = [p for p in self.all('profiles') if p['model_id'] == expected_model['id']]
            if {p['id']: p for p in current} != {p['id']: p for p in expected_profiles}:
                raise ValueError('关联加载配置已变化，请重新检查迁移清单')
            for row in expected_runtimes:
                if self.get('runtimes', row['id']) != row:
                    raise ValueError('关联运行环境已变化，请重新检查迁移清单')
            for row in expected_preparations or []:
                if self.get('ninfer_preparations', row['id']) != row:
                    raise ValueError('模型准备文件已变化，请重新检查迁移清单')
            if self.get('models', new_model['id']):
                raise ValueError('迁移目标已经登记，请刷新模型库')
            self.connection.execute("INSERT INTO records(kind,id,body) VALUES('models',?,?)",
                                    (new_model['id'], json.dumps(new_model, ensure_ascii=False)))
            for row in replacements:
                self.connection.execute("UPDATE records SET body=? WHERE kind='profiles' AND id=?",
                                        (json.dumps(row, ensure_ascii=False), row['id']))
            for row in new_preparations or []:
                self.connection.execute("INSERT INTO records(kind,id,body) VALUES('ninfer_preparations',?,?)",
                                        (row['id'], json.dumps(row, ensure_ascii=False)))
