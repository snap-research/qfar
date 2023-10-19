

import os
import glob

rootfd = '/u6/a/wenzheng/remote2/Snap_project/QRrealcap/QR-code-video-data'
fds = glob.glob('%s/*'%rootfd)
for fd in fds:
    videofds = glob.glob('%s/*.mp4'%fd)
    for videfile in videofds:
        print(videfile)
        # videfile = '/u6/a/wenzheng/tmp3/video1.MOV'
        videopath, _ = os.path.splitext(videfile)
        
        if not os.path.isdir(videopath):
            continue
        if os.path.isdir('%s-detect' % videopath):
            # cmd = 'rm -fr %s-detect' % (videopath,)
            # os.system(cmd)
            continue
        
        cmd = 'python detect.py --source %s --output %s' % (videopath, '%s-detect' % videopath)
        print(cmd)
        os.system(cmd)
        
        
