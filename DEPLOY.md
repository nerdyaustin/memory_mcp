# Deploying the memory_mcp Sync Server

Hosted sync deployment for memory_mcp: local MCP servers keep using local SQLite, then push/pull sessions and memories to a central PostgreSQL-backed FastAPI service.

This deployment uses **no Docker**.

---

## Lab layout

Replace `<APP_IP>`, `<DB_IP>`, and `<DB_PASSWORD>` below with your own values.

| Role | VM | Hostname | IP | Notes |
|---|---:|---|---|---|
| App server | 101 | `memory-mcp` | `<APP_IP>` | FastAPI + systemd service; 4 vCPU, 8GB RAM |
| Database | 102 | `pg2026` | `<DB_IP>` | PostgreSQL 18 + pgvector; 2 vCPU, 4GB RAM |

Set Proxmox CPU type to `host`; NumPy/pgvector client packages may fail on the default `kvm64` CPU type.

```bash
qm set 101 --cpu host
qm set 102 --cpu host
```

Use DHCP reservations or static addresses for both VMs. The app server's `DATABASE_URL` points at the database IP.

---

## 1. PostgreSQL VM (`pg2026`)

Install PostgreSQL 18 and pgvector:

```bash
sudo apt update
sudo apt install -y postgresql-18 postgresql-18-pgvector
sudo systemctl enable --now postgresql
```

Create database and user:

```bash
sudo -u postgres psql -c "CREATE USER memory_mcp WITH PASSWORD '<DB_PASSWORD>';"
sudo -u postgres psql -c "CREATE DATABASE memory_mcp OWNER memory_mcp;"
sudo -u postgres psql -d memory_mcp -c "CREATE EXTENSION IF NOT EXISTS vector;"
sudo -u postgres psql -d memory_mcp -c "GRANT ALL ON SCHEMA public TO memory_mcp;"
```

Allow the app VM to connect:

```bash
echo "host all all <APP_IP>/32 scram-sha-256" | sudo tee -a /etc/postgresql/18/main/pg_hba.conf
sudo sed -i "s/^#listen_addresses =.*/listen_addresses = '*'/" /etc/postgresql/18/main/postgresql.conf
sudo systemctl restart postgresql
```

Verify from `pg2026`:

```bash
PGPASSWORD='<DB_PASSWORD>' psql -U memory_mcp -d memory_mcp -h localhost -c "SELECT version();"
```

---

## 2. App VM (`memory-mcp`)

Install runtime dependencies:

```bash
sudo apt update
sudo apt install -y git python3-pip
sudo pip install --break-system-packages fastapi uvicorn sqlalchemy asyncpg pgvector pydantic
```

Install the repo:

```bash
git clone https://github.com/nerdyaustin/memory_mcp.git /tmp/memory_mcp
sudo rm -rf /opt/memory_mcp
sudo cp -r /tmp/memory_mcp /opt/memory_mcp
sudo chown -R austin:austin /opt/memory_mcp
```

Create the systemd service:

```ini
# /etc/systemd/system/memory-mcp-hosted.service
[Unit]
Description=memory_mcp Sync Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=austin
WorkingDirectory=/opt/memory_mcp
Environment="PYTHONPATH=/opt/memory_mcp"
Environment="DATABASE_URL=postgresql+asyncpg://memory_mcp:<DB_PASSWORD>@<DB_IP>:5432/memory_mcp"
Environment="HOST=0.0.0.0"
Environment="PORT=8000"
ExecStart=/usr/bin/python3 -c "from hosted.server import main; main()"
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Start it:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now memory-mcp-hosted
curl http://localhost:8000/health
```

Expected:

```json
{"status":"ok"}
```

---

## 3. API key seed

Generate a key on the app VM:

```bash
python3 - <<'PY'
import secrets
print(secrets.token_hex(32))
PY
```

Save the plaintext key for client machines:

```bash
sudo install -o austin -g austin -m 600 /dev/null /opt/memory_mcp/hosted/api_key.txt
printf '%s\n' 'PASTE_GENERATED_KEY_HERE' | sudo tee /opt/memory_mcp/hosted/api_key.txt >/dev/null
```

Hash and insert it into PostgreSQL:

```bash
KEY_HASH=$(printf '%s' 'PASTE_GENERATED_KEY_HERE' | sha256sum | cut -d' ' -f1)
sudo -u postgres psql -d memory_mcp -c "
  INSERT INTO users (id, name, api_key_hash, created_at)
  VALUES (gen_random_uuid(), 'austin', '$KEY_HASH', now())
  ON CONFLICT (api_key_hash) DO UPDATE SET name = EXCLUDED.name;
"
```

The server stores only the SHA-256 hash. Clients need the plaintext key.

---

## 4. Verify sync endpoints

Health is public:

```bash
curl http://<APP_IP>:8000/health
```

Authenticated machine registration:

```bash
curl -H "Authorization: Bearer $MEMORY_MCP_SYNC_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"machine_id":"00000000-0000-7000-8000-000000000001","hostname":"test-host"}' \
  http://<APP_IP>:8000/machines/register
```

Expected:

```json
{"registered":true,"machine_id":"00000000-0000-7000-8000-000000000001"}
```

---

## 5. Connect client machines

Set these on every workstation running `memory-mcp`:

```bash
export MEMORY_MCP_SYNC_URL=http://<APP_IP>:8000
export MEMORY_MCP_SYNC_KEY=PASTE_GENERATED_KEY_HERE
```

Restart the MCP server. Each machine keeps its local SQLite database, registers a persistent machine UUID, pushes local sessions/memories, and pulls data from the other machines in the background.

On the first configured sync, pre-existing rows in `~/.memory_mcp/memory.db`
are claimed by that machine's persistent UUID and pushed. You do not need to
copy the SQLite file or rescan unchanged session logs.

Manual sync:

```text
sync_now
```

No sync env vars means local-only mode, matching pre-sync behavior.

---

## 6. Cloudflare Tunnel (optional)

Use Cloudflare Tunnel when this leaves the LAN. The origin stays on `localhost:8000`; no inbound firewall port is needed.

```bash
curl -fsSL https://pkg.cloudflare.com/cloudflare-main.gpg | sudo tee /usr/share/keyrings/cloudflare-main.gpg >/dev/null
echo "deb [signed-by=/usr/share/keyrings/cloudflare-main.gpg] https://pkg.cloudflare.com/cloudflared $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/cloudflared.list
sudo apt update
sudo apt install -y cloudflared
cloudflared tunnel login
cloudflared tunnel create memory-sync
```

`/etc/cloudflared/config.yml`:

```yaml
tunnel: TUNNEL_UUID
credentials-file: /root/.cloudflared/TUNNEL_UUID.json

ingress:
  - hostname: sync.yourdomain.com
    service: http://localhost:8000
  - service: http_status:404
```

```bash
cloudflared tunnel route dns memory-sync sync.yourdomain.com
sudo cloudflared service install
sudo systemctl enable --now cloudflared
```

Then client env becomes:

```bash
export MEMORY_MCP_SYNC_URL=https://sync.yourdomain.com
export MEMORY_MCP_SYNC_KEY=PASTE_GENERATED_KEY_HERE
```

---

## 7. Operations

App logs:

```bash
sudo journalctl -u memory-mcp-hosted -f
```

Database counts:

```bash
sudo -u postgres psql -d memory_mcp -c "
  SELECT
    (SELECT count(*) FROM users) AS users,
    (SELECT count(*) FROM machines) AS machines,
    (SELECT count(*) FROM sessions) AS sessions,
    (SELECT count(*) FROM messages) AS messages,
    (SELECT count(*) FROM memories) AS memories;
"
```

Backup:

```bash
sudo -u postgres pg_dump memory_mcp | gzip > memory_mcp-$(date +%F).sql.gz
```
