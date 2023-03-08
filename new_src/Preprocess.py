import os
from tqdm import tqdm
import hashlib


class Preprocess():
    
    def remove_duplicated_images(directory):
        hashes = set()
        removed = 0
        for filename in tqdm(os.listdir(directory)):
            if filename[-3:] in ['jpg', 'peg']:
                path = os.path.join(directory, filename)
                digest = hashlib.sha1(open(path,'rb').read()).digest()
                if digest not in hashes:
                    hashes.add(digest)
                else:
                    os.remove(path)
                    removed += 1
                
        print(f'Removed {removed} files')

