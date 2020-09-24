import pandas as pd

a = [[0,1,2,3,4],['a','b','c','d','e']]
b = [[0,3,4],['f','g','h']]

# make an empty DataFrame (can do this faster but I'm going slow so you see how it works)
df_a = pd.DataFrame()
df_a['time'] = a[0]
df_a['A'] = a[1]
df_a.set_index('time',inplace=True)

print(df_a)

# same for b (a faster way this time)
df_b = pd.DataFrame({'B':b[1]}, index=b[0])

print(df_b)

# now merge the two Series together (the NaNs are in the right place)
df = pd.merge(df_a, df_b, left_index=True, right_index=True, how='outer')

print(df)