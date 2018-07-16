"""
딕셔너리에서 value 기준으로 정렬 -> 역순 -> 가장 개수 많은 key 가져오기
"""

import operator

x = {0:1, 5:10, 3:5}
sorted_by_value = sorted(x.items(), key=operator.itemgetter(1))
print(sorted_by_value)
sorted_by_value.reverse()
print(sorted_by_value)
print(sorted_by_value[0][0])