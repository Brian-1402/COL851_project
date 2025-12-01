## Env setup
- place the csv files in the current directory
### Via uv
```bash
# install uv if not present
wget -qO- https://astral.sh/uv/install.sh | sh
# create virtual environment
uv sync
```
### Via conda/pip
```bash
# after making relevant conda/pip envs
pip install -r requirements.txt
```

## Part 3
- Assumes files present in current dir:
  - `df_patna_covariates.csv`
```bash
uv run part3_compute.py
uv run part3_plot.py
# Plots made in ./outputs/part3
uv run perfomance_plots.py --file ./outputs/part3/patna_performance.csv --output_dir part3_performance_plots/ --prefix laptop_patna
# Plots made in ./part3_performance_plots
```

## Part 4
- With current dir in pi, run:
```bash
uv run part3_compute.py --output_dir ./pi_outputs/part4
uv run perfomance_plots.py --file ./pi_outputs/part4/patna_performance.csv --output_dir part4_performance_plots/ --prefix pi_patna
# Plots made in ./part4_performance_plots
```

### Pi sync
- to sync entire code to pi, at project dir in local machine, run:
```bash
rsync -avPHz --exclude={'.git','.venv','__pycache__'} ./ pi@pi851:~/COL851_project/
```
#### Get back pi results
- at project dir in local machine, run:
```bash
rsync -avPHz --delete pi@pi851:~/COL851_project/pi_outputs/ ./pi_outputs/
```

#### SSH
```bash
ssh pi@pi851
# or if tailscale inactive, this might work when everyone in phone hotspot
ssh pi@10.234.194.8
```

### Zip cmd
- at project dir in local machine, run:
```bash
zip -r COL851_project.zip . -x '*.git*' -x '*__pycache__*' -x '*.venv*'
```

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
