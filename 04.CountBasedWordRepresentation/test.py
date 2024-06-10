import pandas as pd

# 예시 데이터 프레임 생성
data = {'이름': ['홍길동', '이순신', '강감찬'], '나이': [20, 30, 40]}
df = pd.DataFrame(data, index=['a', 'b', 'c'])
print(df)

# loc 사용 예
print(df.loc['a'])  # 인덱스 'a'에 해당하는 데이터 선택
print(df.loc['a':'b'])  # 인덱스 'a'부터 'b'까지의 데이터 선택

# iloc 사용 예
print(df.iloc[0])  # 첫 번째 행(위치 0)에 해당하는 데이터 선택
print(df.iloc[0:2])  # 위치 0부터 1까지의 데이터 선택
