# tasks.ps1 —— 常用命令入口。CI 与本地共用 scripts/run_all_checks.py 里的定义。
#
# 刻意写得**扁平**：没有 switch、没有 helper 函数、不往命令传数组。
# 第一版用了 switch + 一个接收数组的 helper，PowerShell 解析器一直报
# "Missing statement block in switch statement clause"，而错误位置随着行数平移
# （"lock 是关键字"和"$args 是自动变量"两个假设都被证伪）。
# 这是一个便利包装，不值得为它跟解析器较劲 —— 改成最朴素的 if/elseif。
#
# 用法：
#   powershell -File tasks.ps1 test       只跑测试
#   powershell -File tasks.ps1 verify     全部检查（约 15 分钟）
#   powershell -File tasks.ps1 quick      快速版
#   powershell -File tasks.ps1 deps       重新生成并校验锁文件
#   powershell -File tasks.ps1 serve      起平台（默认 8077 端口）
#   powershell -File tasks.ps1 list       只列检查计划

param(
    [string]$Task = 'help',
    [int]$Port = 8077,
    [switch]$NoReset
)

Set-Location -LiteralPath $PSScriptRoot

# 统一 UTF-8：不设这两项，中文与 JSON 会按 ANSI 解码成乱码
chcp 65001 > $null
$env:PYTHONIOENCODING = 'utf-8'
# uv 的缓存默认在 ~/.cache/uv，受限环境里不可写；指到项目内
$env:UV_CACHE_DIR = Join-Path $PSScriptRoot 'build\.uv-cache'

$py = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path $py)) { $py = Join-Path $PSScriptRoot '.venv/bin/python' }
if (-not (Test-Path $py)) {
    Write-Host '找不到 .venv 里的 python。先建一个：python -m venv .venv，再 pip install -r requirements.txt'
    exit 1
}

if ($Task -eq 'setup') {
    & $py -m pip install -r requirements.txt
    & $py scripts/lock_requirements.py --check
}
elseif ($Task -eq 'test') {
    # 不要再加 -q：pyproject 的 addopts 里已经有一个，两个会变成 -qq 把汇总行吞掉
    & $py -m pytest tests/
}
elseif ($Task -eq 'verify') {
    & $py scripts/run_all_checks.py
}
elseif ($Task -eq 'quick') {
    & $py scripts/run_all_checks.py --quick
}
elseif ($Task -eq 'list') {
    & $py scripts/run_all_checks.py --list
}
elseif ($Task -eq 'deps') {
    & $py scripts/lock_requirements.py
    & $py scripts/lock_requirements.py --check
}
elseif ($Task -eq 'warehouse') {
    & $py scripts/run_warehouse.py
}
elseif ($Task -eq 'serve') {
    # m5/m6 的数仓段落要 build/warehouse.duckdb 存在，否则会静默跳过
    if (-not (Test-Path 'build\warehouse.duckdb')) {
        Write-Host '数仓不存在，先建一个（约 10 秒）'
        & $py scripts/run_warehouse.py
    }
    if ($NoReset) {
        & $py scripts/run_platform.py --port $Port
    }
    else {
        & $py scripts/run_platform.py --port $Port --reset
    }
}
else {
    Write-Host '可用命令：setup / test / verify / quick / list / deps / warehouse / serve'
    Write-Host '例如：powershell -File tasks.ps1 verify'
}

if ($LASTEXITCODE -ne 0) {
    Write-Host "[失败] 退出码 $LASTEXITCODE"
    exit $LASTEXITCODE
}
exit 0
