def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area) / 2.0
    return area


corners = [(3.0, 4.0), (5.0, 11.0), (12.0, 8.0), (9.0, 5.0), (5.0, 6.0)]
print(PolygonArea(corners))
