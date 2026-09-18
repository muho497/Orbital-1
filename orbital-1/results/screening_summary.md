# CLIP screening summary

model ViT-L-14/openai

## output

| class | n | agree % | hard-fail % | drift % | retained % | top confusion | top hard mode | top drift |
|---|---|---|---|---|---|---|---|---|
| crater | 1000 | 31.2 | 33.7 | 30.8 | 29.4 | perforation (401) | person (179) | device (174) |
| perforation | 800 | 76.8 | 5.8 | 8.0 | 76.6 | melt_splash (35) | moon_terrain (30) | plain_texture (53) |
| petaling | 600 | 71.7 | 7.0 | 11.0 | 69.3 | perforation (48) | person (18) | device (36) |
| spallation | 600 | 5.7 | 30.2 | 49.7 | 5.2 | debris_spray (126) | moon_terrain (141) | plain_texture (278) |
| crack_web | 800 | 24.0 | 28.4 | 44.6 | 22.8 | solar_damage (166) | moon_terrain (154) | plain_texture (312) |
| mli_damage | 1000 | 55.4 | 13.2 | 19.0 | 55.2 | nominal (150) | person (83) | plain_texture (101) |
| solar_damage | 1000 | 81.7 | 1.2 | 6.0 | 81.6 | crack_web (56) | moon_terrain (7) | plain_texture (58) |
| debris_spray | 600 | 34.8 | 22.2 | 33.2 | 32.2 | crater (136) | moon_terrain (100) | plain_texture (176) |
| cfrp_delamination | 600 | 27.5 | 31.0 | 65.8 | 26.2 | melt_splash (183) | moon_terrain (86) | plain_texture (377) |
| microcrater_field | 800 | 5.4 | 52.2 | 66.0 | 4.6 | debris_spray (326) | moon_terrain (372) | plain_texture (505) |
| melt_splash | 500 | 12.8 | 40.4 | 45.0 | 11.6 | debris_spray (91) | moon_terrain (148) | plain_texture (162) |
| structural_severe | 400 | 50.7 | 15.8 | 21.2 | 50.7 | nominal (53) | person (41) | plain_texture (39) |
| nominal | 1500 | 47.6 | 9.2 | 7.8 | 46.4 | perforation (305) | cartoon (52) | device (65) |
| _all | 10200 | 42.7 | 20.8 | 28.3 | 41.6 |  (0) |  (0) |  (0) |

## output_v2

| class | n | agree % | hard-fail % | drift % | retained % | top confusion | top hard mode | top drift |
|---|---|---|---|---|---|---|---|---|
| crater | 1000 | 7.6 | 68.1 | 75.6 | 7.4 | melt_splash (390) | moon_terrain (581) | plain_texture (749) |
| perforation | 800 | 42.9 | 20.1 | 40.5 | 42.4 | melt_splash (124) | cartoon (117) | plain_texture (299) |
| petaling | 600 | 96.7 | 61.7 | 3.7 | 35.8 | melt_splash (6) | flower (361) | plain_texture (22) |
| spallation | 600 | 6.7 | 35.0 | 89.3 | 5.8 | melt_splash (126) | cartoon (115) | plain_texture (535) |
| crack_web | 800 | 37.0 | 10.0 | 34.5 | 36.6 | solar_damage (265) | moon_terrain (42) | plain_texture (275) |
| mli_damage | 1000 | 60.4 | 9.0 | 27.0 | 60.1 | petaling (142) | cartoon (65) | plain_texture (258) |
| solar_damage | 327 | 55.7 | 3.7 | 22.0 | 55.7 | crack_web (67) | moon_terrain (8) | plain_texture (72) |
| _all | 5127 | 41.4 | 31.3 | 44.0 | 33.9 |  (0) |  (0) |  (0) |
