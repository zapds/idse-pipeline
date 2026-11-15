# Deployment: smog.zapdos.me

- VPS: `oracle-spend` (Oracle, aarch64). Project dir `/home/ubuntu/smog-forecast`
  - `releases/<timestamp>/` — one directory per release; `current` -> active release
  - `venv/` — shared Python 3.12 venv built from `deploy/requirements-web.txt` (versions must match the
    ones the models were trained with, or joblib loading can fail)
- PM2 app `smog-forecast` (gunicorn, 2 workers) on `127.0.0.1:3107`; config `deploy/ecosystem.config.cjs`
- Caddy site block `smog.zapdos.me` in `/etc/caddy/Caddyfile` -> `reverse_proxy 127.0.0.1:3107`
- DNS: Cloudflare (proxied)

## Files shipped (runtime only)
`src/`, `web/`, `models/`, `reports/{figures,metrics,report.pdf,report.md}`, `notebooks/*.ipynb`,
`data/processed/daily_inputs.parquet`, `data/interim/*_cleaning_log.json`, `deploy/`.

## Update
1. `./run_all.sh` locally (or at least `pytest`), then copy the files above into a new
   `releases/<timestamp>` with rsync (no `--delete` on shared paths).
2. `ln -sfn releases/<timestamp> current`, then
   `pm2 startOrReload /home/ubuntu/smog-forecast/current/deploy/ecosystem.config.cjs --only smog-forecast --update-env && pm2 save`.
3. Rollback: point `current` back at the previous release and reload the same way.
