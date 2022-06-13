import time
import rich.progress

for n in rich.progress.track(range(5)):
    # rich.print('foo')
    # rich.print('')
    rich.print()
    time.sleep(0.5)