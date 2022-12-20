def get_id(filename):
    #SUBJECT_ENCOUNTER_DEVICE_CAPTURE_RESOLUTION_FGRP.EXT (SD301) or
    #SUBJECT_DEVICE_RESOLUTION_CAPTURE_FRGP.EXT (SD302)
    return filename.split('_')[0]

def get_fgrp(filename):
    filename = filename[:-4]
    return filename.split('_')[-1]
    #return filename.replace('.', '_').split('_')[-2]

def get_sensor(filename):
    tokens = filename.split('_')
    # SD301 version
    if tokens[1].isnumeric():
        return tokens[1] + '_' + tokens[2] #SUBJECT_ENCOUNTER_DEVICE_CAPTURE_RESOLUTION_FRGP.EXT
    # SD302 version
    return tokens[1] #SUBJECT_DEVICE_RESOLUTION_CAPTURE_FRGP.EXT
    #return filename.split('_')[1]#'_'.join(filename.split('_')[1:2+1])
