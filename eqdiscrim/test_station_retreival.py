import eqdiscrim_io as io

PF_filename = 'PF_stations.txt'

url = 'http://ws.resif.fr/fdsnws/station/1/query?network=PF&channel=???&format=text&level=channel'
io.get_RESIF_info(url, PF_filename)

X, short_list = io.read_sta_file(PF_filename)
sta_dict = io.create_station_objects(X, short_list)

sta_list = sta_dict.keys()
sta_list.sort()
for sta in sta_list:
    print sta_dict[sta]

