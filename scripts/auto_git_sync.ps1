# scripts/auto_git_sync.ps1
# Sincronización automática de código, pesos del modelo y métricas con GitHub

param (
    [string]$CommitMessage = "chore: auto-sync project updates, model weights and results"
)

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "   Sincronización Automática con GitHub   " -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan

# 1. Staging de archivos del proyecto (código, pesos, métricas, gráficas, docs y dataset procesado)
Write-Host "[1/3] Añadiendo cambios de código, modelos, dataset y resultados..." -ForegroundColor Yellow
git add src/ config/ models/ results/ docs/ scripts/ README.md .gitignore dataset/train/ dataset/val/ dataset/test/

# 2. Comprobar si hay cambios para commitear
$changes = git status --porcelain --untracked-files=no
if ($changes) {
    Write-Host "[2/3] Creando commit: $CommitMessage" -ForegroundColor Yellow
    git commit -m "$CommitMessage"
    
    # 3. Push a main
    Write-Host "[3/3] Subiendo cambios a origin/main..." -ForegroundColor Yellow
    git push origin main
    if ($LASTEXITCODE -eq 0) {
        Write-Host " Sincronización completada con éxito." -ForegroundColor Green
    } else {
        Write-Host " Error al subir cambios a GitHub. Verifica tu conexión o credenciales." -ForegroundColor Red
    }
} else {
    Write-Host " No hay cambios pendientes en código, modelos o resultados." -ForegroundColor Green
}
