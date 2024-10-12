


# Function to parse the file to extract MD5 and filenames
def extract_md5_from_details(md5_file):
    md5s = {}
    with open(md5_file, "r") as f:
        lines = f.readlines()
        current_file = None
        for line in lines:
            if line.startswith("gs://"):
                current_file = line.strip().rstrip(":")
                current_file = current_file.split("/")[-1]
            if "Hash (md5)" in line:
                md5s[current_file] = line.split(":")[1].strip()
    return md5s


# Function to parse the file to extract MD5 and filenames
def extract_md5_from_details2(md5_file):
    md5s = {}
    with open(md5_file, "r") as f:
        lines = f.readlines()
        current_file = None
        for line in lines:
            if line.startswith("Hashes [hex]"):
                current_file = line.strip().rstrip(":")
                current_file = current_file.split("/")[-1]
            if "Hash (md5)" in line:
                md5s[current_file] = line.split(":")[1].strip()
    return md5s



# Function to parse the file to extract crc32c and filenames
def extract_crc32c_from_details2(md5_file):
    crcs = {}
    with open(md5_file, "r") as f:
        lines = f.readlines()
        current_file = None
        for line in lines:
            if line.startswith("Hashes [hex]"):
                current_file = line.strip().rstrip(":")
                current_file = current_file.split("/")[-1]
            if "Hash (crc32c)" in line:
                crcs[current_file] = line.split(":")[1].strip()
    return crcs



# Function to parse the file to extract crc32c and filenames
def extract_hashes_from_gcloudstorage(source_hash):

    crcs = {}
    md5s = {}

    with open(source_hash, "r") as f:
        lines = f.readlines()
        current_file = None
        for line in lines:
            
            if line.startswith("crc32c_hash:"):
                curr_crc =  line.split(":")[1].strip()

            elif line.startswith("md5_hash:"):
                curr_md5 =  line.split(":")[1].strip()

            elif line.startswith("url:"):
                current_file = line.split("/")[-1].strip()
                crcs[current_file] = curr_crc
                md5s[current_file] = curr_md5
            # else:
            #     print(f'cruff:{line.strip()}')


    return crcs, md5s



# Function to parse the file to extract crc32c and filenames
def extract_hashes_from_gsutil(source_hash):

    crcs = {}
    md5s = {}

    with open(source_hash, "r") as f:
        lines = f.readlines()
        current_file = None
        for line in lines:
            if line.startswith("Hashes [hex]"):
                current_file = line.strip().rstrip(":")
                current_file = current_file.split("/")[-1]
            if "Hash (crc32c)" in line:
                crcs[current_file] = line.split(":")[1].strip()
            if "Hash (md5)" in line:
                md5s[current_file] = line.split(":")[1].strip()

    return crcs, md5s
