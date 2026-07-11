# Build the submission image and report its size. Run on a machine with Docker.
# The registry limit is 10 GB *compressed*; `docker images` shows the larger
# uncompressed size, so treat it as an upper bound. The lean image (python-slim,
# deterministic solvers + Fireworks, no bundled model) is well under the limit.

$ErrorActionPreference = "Stop"
$IMAGE = "token-efficient-agent"

Write-Host "Building $IMAGE (lean: deterministic solvers + Fireworks, no bundled model)..." -ForegroundColor Cyan
docker build -t $IMAGE .

Write-Host "`nUncompressed image size:" -ForegroundColor Cyan
docker images $IMAGE --format "{{.Repository}}:{{.Tag}}  {{.Size}}"

Write-Host "`nMeasuring compressed size (this is what the 10 GB limit checks)..." -ForegroundColor Cyan
docker save $IMAGE -o image.tar
$gz = "image.tar.gz"
& { param($in,$out)
    $fsIn = [IO.File]::OpenRead($in)
    $fsOut = [IO.File]::Create($out)
    $gzip = New-Object IO.Compression.GZipStream($fsOut, [IO.Compression.CompressionMode]::Compress)
    $fsIn.CopyTo($gzip); $gzip.Close(); $fsOut.Close(); $fsIn.Close()
} "image.tar" $gz
$sizeGB = [math]::Round((Get-Item $gz).Length / 1GB, 2)
Write-Host "compressed image: $sizeGB GB (limit 10 GB)" -ForegroundColor Green
Remove-Item image.tar, $gz -ErrorAction SilentlyContinue

Write-Host "`nLocal test run (fill in dev creds):" -ForegroundColor Cyan
Write-Host '  docker run --rm ``'
Write-Host '    -e FIREWORKS_API_KEY=<key> -e FIREWORKS_BASE_URL=<url> -e ALLOWED_MODELS=<ids> ``'
Write-Host '    -v ${PWD}/data/input:/input -v ${PWD}/data/output:/output ``'
Write-Host "    $IMAGE"
