import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/huy/Desktop/Intervention-project_Vehicle-Manipulator-Systems-haadi_experimental/install/localization'
