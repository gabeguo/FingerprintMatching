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
For UB database - successfully tested and debugged
"""

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
    device = sessionId + 'UB'
    resolution = 'NaN'
    capture = 'NaN'
    fgrp = convert_hand_finger_to_fgrp(handId, fingerId)

    return '{}_{}_{}_{}_{}.{}'.format(subject, device, resolution, capture, fgrp, the_ext)
# Go from <SessionID>_<DeviceName>_<IdentityID>_<BackgroundID>_<HandID>_image_fingerprintRandomseq_(<Confidence>)_<FingerID> (UB format)
# To SUBJECT_DEVICE_RESOLUTION_CAPTURE_FRGP.EXT (NIST SD302 format)
def rename_ub_contactless_sample(filename):
    filename_without_ext, the_ext = filename.rsplit('.', 1)
    assert len(the_ext) == 3
    
    tokens = filename_without_ext.split('_')

    sessionId = tokens[0]
    deviceName = tokens[1]
    identityId = tokens[2]
    backgroundId = tokens[3]
    handId = tokens[4]
    imageStr = tokens[5]
    fingerprintRandomSeq = tokens[6]
    fingerId = tokens[-1]
    assert '.' not in fingerId

    subject = identityId
    device = sessionId + deviceName + backgroundId
    resolution = 'NaN'
    capture = fingerprintRandomSeq
    fgrp = convert_hand_finger_to_fgrp(handId, fingerId)

    return '{}_{}_{}_{}_{}.{}'.format(subject, device, resolution, capture, fgrp, the_ext)

def convert_hand_finger_to_fgrp(handId, fingerId):
    handId = handId.lower()
    fingerId = fingerId.lower()
    if handId == 'right':
        if fingerId in ['thumb']:
            return '01'
        if fingerId in ['index', '0']:
            return '02'
        elif fingerId in ['middle', '1']:
            return '03'
        elif fingerId in ['ring', '2']:
            return '04'
        elif fingerId in ['little', '3']:
            return '05'
        else:
            raise ValueError('invalid fingerId in UB database')
    elif handId == 'left':
        if fingerId in ['thumb']:
            return '06'
        if fingerId in ['index', '0']:
            return '07'
        elif fingerId in ['middle', '1']:
            return '08'
        elif fingerId in ['ring', '2']:
            return '09'
        elif fingerId in ['little', '3']:
            return '10'
        else:
            raise ValueError('invalid fingerId in UB database')
    else:
        raise ValueError('invalid handId in UB database')

"""
For SOCOFing database
Has been verified to be correct (tests)
"""

REAL_SOCOFING = 'Real'
EASY_SOCOFING = 'Altered-Easy'
MEDIUM_SOCOFING = 'Altered-Medium'
HARD_SOCOFING = 'Altered-Hard'

def rename_socofing_file(filename, alterationDifficulty):
    assert '/' not in filename
    filename = filename.lower() # convert to lowercase
    if alterationDifficulty == REAL_SOCOFING:
        return rename_unaltered_socofing(filename)
    elif alterationDifficulty in [EASY_SOCOFING, MEDIUM_SOCOFING, HARD_SOCOFING]:
        return rename_altered_socofing(filename, alterationDifficulty)
    else:
        raise ValueError('invalid alteration for socofing: {}'.format(alterationDifficulty))

# converts from SUBJECT__GENDER_HAND_FINGERNAME_finger.BMP
# to SUBJECT_DEVICE_RESOLUTION_CAPTURE_FRGP.EXT (NIST SD302 format)
def rename_unaltered_socofing(filename):
    filename_without_ext, the_ext = filename.rsplit('.', 1)
    assert len(the_ext) == 3
    the_ext = the_ext.lower()

    subject, dummy, gender, hand, fingername, fingerStr = filename_without_ext.split('_')

    assert dummy == ''
    assert subject != ''
    assert gender in ['m', 'f']

    subject = subject
    device = 'Real'
    resolution = 'NaN'
    capture = 'NaN'
    fgrp = convert_hand_finger_to_fgrp(handId=hand, fingerId=fingername)

    return '{}_{}_{}_{}_{}.{}'.format(subject, device, resolution, capture, fgrp, the_ext)

# converts from SUBJECT__GENDER_HAND_FINGERNAME_finger_ALTERATION.BMP
# to SUBJECT_DEVICE_RESOLUTION_CAPTURE_FRGP.EXT (NIST SD302 format)
def rename_altered_socofing(filename, alterationDifficulty):
    filename_without_ext, the_ext = filename.rsplit('.', 1)
    assert len(the_ext) == 3
    the_ext = the_ext.lower()

    subject, dummy, gender, hand, fingername, fingerStr, alterationType = filename_without_ext.split('_')

    assert dummy == ''
    assert subject != ''
    assert gender in ['m', 'f']

    subject = subject
    device = alterationDifficulty + alterationType.upper()
    resolution = 'NaN'
    capture = 'NaN'
    fgrp = convert_hand_finger_to_fgrp(handId=hand, fingerId=fingername)

    return '{}_{}_{}_{}_{}.{}'.format(subject, device, resolution, capture, fgrp, the_ext)