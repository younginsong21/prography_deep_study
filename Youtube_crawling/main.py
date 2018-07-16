# -*- coding: utf-8 -*-

from youtube_search_and_download import YouTubeHandler

search_key = '철권7 영상'  # keywords
yy = YouTubeHandler(search_key)
yy.download_as_audio = 0  # 1- download as audio format, 0 - download as video
yy.set_num_playlist_to_extract(10)  # number of playlist to download

print 'Get all the playlist'
yy.get_playlist_url_list()
print yy.playlist_url_list

## Get all the individual video and title from each of the playlist
yy.get_video_link_fr_all_playlist()
for key in yy.video_link_title_dict.keys():
    print key, '  ', yy.video_link_title_dict[key]
    print
print

print 'download video'
yy.download_all_videos(dl_limit=50)  # number of videos to download.