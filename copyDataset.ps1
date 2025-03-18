
# 1. Define variables
$localPath = "./asl_alphabet_train"  
$podName = "asl-decoder"           
$namespace       = "anthony-lab"          
$destinationPath = "/pvc-files"         


$folders = Get-ChildItem -Directory -Path $localPath

foreach ($folder in $folders) {
    $folderName = $folder.Name

    $localFolderPath = $folder.FullName

    $destFolderPath = "$destinationPath/$folderName"

    Write-Host "Copying folder '$folderName' to '$podName':$destFolderPath ..."

    try {
        & kubectl cp $localFolderPath "'$podName':$destFolderPath" -n $namespace
        Write-Host "Copied '$folderName' successfully."
    } catch {
        Write-Error "Error copying '$folderName': $_"
    }
}

Write-Host "All done!"
