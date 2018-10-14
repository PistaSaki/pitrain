def concat_dics(*dics):
    dic = dict()
    for d in dics:
        if d is not None: 
            dic.update(d)
    return dic
    
if __name__ == "__main__":
    print( 
        'concat_dics({"a":1}, {1:"a"}) =', 
        concat_dics({"a":1}, {1:"a"})
    )

def BFGS_update_B(B0, x1, x0, df_x1, df_x0, check_positive = True):
    y = (df_x1 - df_x0).reshape([-1, 1])
    s = (x1 - x0).reshape([-1, 1])
    
    if check_positive:
        assert y.T @ s > 0
    
    B1 = B0 + (y @ y.T) / (y.T @ s) - (B0 @ s ) @ (s.T @ B0) / (s.T @ B0 @ s)
    return B1
    