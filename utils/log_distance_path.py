def calc_log_distance_path(rss, a, n):
    '''
    calc_distance_log_path: 
        Parameter:
            - rss   : Mean RSSI Value
            - a     : Reference Power
            - n     : Path Loss Exponent (def: 3)
    '''
    return pow(10, ((rss-a)/(-10*n)))