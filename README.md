## Part 2 services

### Prometheus
- To run: `prometheus --config.file=prometheus.yml`
- Main dashboard: http://localhost:9090

#### Windows exporter
- Runs at http://localhost:9182/metrics
- Runs as a windows service. powershell commands to start/stop/restart:
    ```powershell
    Get-Service windows_exporter
    Start-Service windows_exporter
    Stop-Service windows_exporter
    Restart-Service windows_exporter
    ```

- Installation command used:
    ```powershell
    $PSNativeCommandArgumentPassing = 'Legacy'
    msiexec /i "C:\Users\brian\Downloads\windows_exporter-0.31.3-amd64.msi" --% ADDLOCAL=FirewallException ENABLED_COLLECTORS="[defaults],cache,gpu,cpu_info,diskdrive,pagefile,performancecounter,process,textfile"
    ```


### Grafana
- Runs at http://localhost:3000 (admin/grafanaForNow)
- Windows service name: `grafana`

### Pi sync
- at project dir in local machine, run:
```bash
rsync -avPHz --exclude={'.git','.venv','__pycache__'} ./ pi@pi851:~/COL851_project/
```
#### Get back pi results
- at project dir in local machine, run:
```bash
rsync -avPHz pi@pi851:~/COL851_project/outputs/ ./pi_outputs/
```

#### SSH
```bash
ssh pi@pi851
# or if tailscale inactive, this might work when everyone in phone hotspot
ssh pi@10.234.194.8
```

### Zip code
- at project dir in local machine, run:
```bash
zip -r COL851_project.zip . -x '*.git*' -x '*__pycache__*' -x '*.venv*'
```