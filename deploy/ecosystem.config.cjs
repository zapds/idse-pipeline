// PM2 config for the inference website on the VPS (smog.zapdos.me -> Caddy -> 127.0.0.1:3107).
// /home/ubuntu/smog-forecast/current is a symlink to the active release; the venv is shared.
const BASE = "/home/ubuntu/smog-forecast";
module.exports = {
  apps: [{
    name: "smog-forecast",
    cwd: `${BASE}/current/web`,
    script: `${BASE}/venv/bin/gunicorn`,
    args: "-w 2 -b 127.0.0.1:3107 --timeout 60 --access-logfile - app:app",
    interpreter: "none",
    env: { PYTHONUNBUFFERED: "1", OMP_NUM_THREADS: "1" },
    autorestart: true,
    max_restarts: 10,
    max_memory_restart: "900M",
  }],
};
