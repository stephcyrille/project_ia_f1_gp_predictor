# Function to convert time in 'mm:ss.sss' format to milliseconds
def time_to_milliseconds(time_str:str) -> int:
    minutes, seconds = map(float, time_str.split(':'))
    return int((minutes * 60 + seconds) * 1000)