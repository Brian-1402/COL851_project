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
