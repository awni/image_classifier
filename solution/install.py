import subprocess
import os.path
import os

osName = os.name
isWindows = os.name == 'nt'

BOTH = 'both'
ONLY_NUMPY = 'numpy'
NEITHER = 'neither'

def testNumpy():
    try:
        import numpy
        return True
    except:
        return False

def testPIL():
    try:
        import PIL
        return True
    except:
        return False

def testImports():
    numpy = testNumpy()
    PIL = testPIL()
    if numpy and PIL: return BOTH
    if numpy: return NUMPY_ONLY
    return NEITHER

def installNumpy(installerPath):
    print '(1/ 2) Installing numpy:'
    if isWindows:
        subprocess.call(['python', installerPath, 'install', 'numpy==1.6.2'])
    else:
        print 'You may have to log in. If prompted, enter the password for this computer.'
        subprocess.call(['sudo', 'python', installerPath, 'install', 'numpy==1.6.2'])
    print ''

def installPIL(installerPath):
    print '(1/ 2) Installing PIL:'
    if isWindows:
        subprocess.call(['python', installerPath, 'install', 'PIL'])
    else:
        print 'You may have to log in. If prompted, enter the password for this computer.'
        subprocess.call(['sudo', 'python', installerPath, 'install', 'PIL'])
    print ''

def pipInstall(status):
    pluginDir = 'plugins'
    pipRootDir = os.path.join(pluginDir, 'pip')
    pipDir = os.path.join(pipRootDir, 'pip')
    installPath = os.path.join(pipDir, 'runner.py')
    if status == NEITHER:
        installNumpy(installPath)
    installPIL(installPath)


def run():
    print '-----------------------------------------'
    print 'Welcome to the CS221 installer.'
    print 'In order to make your project easier to program'
    print 'we are going to use some basic python tools:'
    print 'numpy and PIL'
    print '-----------------------------------------'
    print ''

    status = testImports()

    if status == BOTH:
        print 'It looks like you have everything installed.'
        print 'Good luck with the assignment.'
        return

    if status == ONLY_NUMPY:
        print 'It looks like you only have numpy installed.'
        print 'We are going to try and install PIL.'
        print 'But know that this is optional to the assignment.'
    raw_input("Press Enter to continue:")
    print ''

    pipInstall(status)

    status = testImports()

    print '-----------------------------------------'
    if status == BOTH:
        print 'It looks like you have everything installed.'
        print 'Good luck with the assignment.'
    if status == ONLY_NUMPY:
        print 'It looks like we successfully installed numpy,'
        print 'but were unable to install PIL.'
        print 'PIL is optional for the assignment and is'
        print 'available on the corn machines.'
    if status == NEITHER:
        print 'Oh no! We were not able to install numpy OR'
        print 'PIL (numpy is essential). Email the'
        print 'transcript of this run to the staff list right now,'
        print 'and we will help you figure out how to move forward.'
        print 'In the mean time you can use the corn machines.'
    print '-----------------------------------------'

if __name__ == "__main__":
    run()
