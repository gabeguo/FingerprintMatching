"""
For NIST databases
"""

def get_id(filename):
    assert '/' not in filename
    #SUBJECT_ENCOUNTER_DEVICE_CAPTURE_RESOLUTION_FGRP.EXT (SD301) or
    #SUBJECT_DEVICE_RESOLUTION_CAPTURE_FRGP.EXT (SD302)
    return filename.split('_')[0]

def get_fgrp(filename):
    assert '/' not in filename
    assert filename[-4] == '.'
    filename = filename[:-4]
    return filename.split('_')[-1]

def get_sensor(filename):
    assert '/' not in filename # want filename, not filepath
    tokens = filename.split('_')
    # SD301 version
    if tokens[1].isnumeric():
        return tokens[1] + '_' + tokens[2] #SUBJECT_ENCOUNTER_DEVICE_CAPTURE_RESOLUTION_FRGP.EXT
    # SD302 (&SD300) version
    return tokens[1] #SUBJECT_DEVICE_RESOLUTION_CAPTURE_FRGP.EXT for SD302, SUBJECT_IMPRESSION_PPI_FRGP.EXT for SD300
    #return filename.split('_')[1]#'_'.join(filename.split('_')[1:2+1])

"""
For UB database
"""

# TODO: test this code

def rename_ridgebase_file(filename):
    assert '/' not in filename # should only be filename, not path
    if is_ub_contactbased_sample(filename):
        return rename_ub_contactbased_sample(filename)
    return rename_ub_contactless_sample(filename)

def is_ub_contactbased_sample(filename):
    return len(filename.split('_')) == 4

# Go from <SessionID>_<IdentityID>_<HandID>_<FingerID>.EXT (UB format)
# To SUBJECT_DEVICE_RESOLUTION_CAPTURE_FRGP.EXT (NIST SD302 format)
def rename_ub_contactbased_sample(filename):
    filename_without_ext, the_ext = filename.split('.')
    assert len(the_ext) == 3
    sessionId, identityId, handId, fingerId = filename_without_ext.split('_')
    
    subject = identityId
    device = sessionId
    resolution = 'NaN'
    capture = 'NaN'
    fgrp = convert_hand_finger_to_fgrp(handId, fingerId)

    return '{}_{}_{}_{}_{}.{}'.format(subject, device, resolution, capture, fgrp, the_ext)
# Go from <SessionID>_<DeviceName>_<IdentityID>_<BackgroundID>_<HandID>_image_fingerprintRandomseq_(<Confidence>)_<FingerID> (UB format)
# To SUBJECT_DEVICE_RESOLUTION_CAPTURE_FRGP.EXT (NIST SD302 format)
def rename_ub_contactless_sample(filename):
    filename_without_ext, the_ext = filename.split('.')
    assert len(the_ext) == 3
    
    tokens = filename_without_ext.split('_')

    sessionId = tokens[0]
    deviceName = tokens[1]
    identityId = tokens[2]
    backgroundId = tokens[3]
    handId = tokens[4]
    fingerId = tokens[-1]
    assert '.' not in fingerId

    subject = identityId
    device = sessionId + deviceName + backgroundId
    resolution = 'NaN'
    capture = 'NaN'
    fgrp = convert_hand_finger_to_fgrp(handId, fingerId)

    return '{}_{}_{}_{}_{}.{}'.format(subject, device, resolution, capture, fgrp, the_ext)

def convert_hand_finger_to_fgrp(handId, fingerId):
    if handId == 'Right':
        if fingerId == 'Index' or '0':
            return '02'
        elif fingerId == 'Middle' or '1':
            return '03'
        elif fingerId == 'Ring' or '2':
            return '04'
        elif fingerId == 'Little' or '3':
            return '05'
        else:
            raise ValueError('invalid fingerId in UB database')
    elif handId == 'Left':
        if fingerId == 'Index' or '0':
            return '07'
        elif fingerId == 'Middle' or '1':
            return '08'
        elif fingerId == 'Ring' or '2':
            return '09'
        elif fingerId == 'Little' or '3':
            return '10'
        else:
            raise ValueError('invalid fingerId in UB database')
    else:
        raise ValueError('invalid handId in UB database')