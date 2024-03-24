import os
import zipfile
import gdown

_TARGET_DIR = os.path.expanduser('.') #Directory to store data
_FILE_ID= "1368NySV1bWwlOhp525TD0HJEvbCG0WZr" #Use Id for zip file on drive, ensure "Anyone with link can access"


def save_with_gdown(id, destination):
    url = 'https://drive.google.com/uc?id='+id
    gdown.download(url, destination, quiet=False)  



if __name__ == '__main__':      
    zip_path = _TARGET_DIR + '/CityData.zip'
    os.remove(zip_path) if os.path.exists(zip_path) else None
    save_with_gdown(_FILE_ID, zip_path)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(_TARGET_DIR)
        os.remove(zip_path)
        
    except zipfile.BadZipFile:
        print('Not a zip file or a corrupted zip file')
