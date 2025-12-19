import zipfile

def build_sb3(json_path="project.json", output_path="output.sb3"):
    with zipfile.ZipFile(output_path, "w") as z:
        z.write(json_path, arcname="project.json")
    return output_path
