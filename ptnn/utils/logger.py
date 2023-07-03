
def update_loading_bar(progress, total, bar_length=40):
    percent = progress / total
    filled_length = int(bar_length * percent)
    bar = '=' * filled_length + '-' * (bar_length - filled_length)
    print(f'[{bar}] {progress}/{total} ({int(percent * 100)}%)', end='\r')
