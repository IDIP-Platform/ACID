
def imagej_compatible_metadata_dict(metadata_dict:dict):
    imagej_compatible_dict ={}
    for meta_data in metadata_dict:
        imagej_compatible_dict[f"custom_{meta_data}"]=metadata_dict[meta_data]
    return imagej_compatible_dict
