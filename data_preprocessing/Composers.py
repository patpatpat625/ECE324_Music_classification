import string
import selenium
from selenium import webdriver
import selenium.webdriver.support.ui as ui
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import requests
import csv
import time
from pathlib import Path

webpages = ['https://imslp.org/wiki/Category:Bach,_Johann_Sebastian',
            'https://imslp.org/wiki/Category:Vivaldi%2C_Antonio',
            'https://imslp.org/wiki/Category:Mozart%2C_Wolfgang_Amadeus',
            'https://imslp.org/wiki/Category:Beethoven%2C_Ludwig_van',
            'https://imslp.org/wiki/Category:Debussy%2C_Claude',
            'https://imslp.org/wiki/Category:Chopin%2C_Fr%C3%A9d%C3%A9ric',
            'https://imslp.org/wiki/Category:Brahms%2C_Johannes',
            'https://imslp.org/wiki/Category:Tchaikovsky%2C_Pyotr']
driver = webdriver.Chrome()
bad_link_list = ["O Traurigkeit, o Herzeleid, BWV Anh.200",
                 "Prelude and Fugue in C major, BWV 545",
                 "Vom Himmel hoch, da komm ich her, BWV 769",
                 "Così fan tutte, K.588",
                 "Don Giovanni, K.527",
                 "Flute Concerto in G major, K.313/285c",
                 "Piano Concerto No.10 in E-flat major, K.365/316a",
                 "Le nozze di Figaro, K.492",
                 "5 Variations on 'Rule Britannia', WoO 79",
                 "Mazurka in B-flat major 'Wołowska', B.73",
                 "Le promenoir des deux amants"]

Music_ID = 0
label_csv = open("Music_labels.csv", 'w')
writer = csv.writer(label_csv)

header = ['Music_ID', 'Name of Work', 'Composer', 'Period']
writer.writerow(header)
for composer_link in webpages:
    driver.get(composer_link)
    composer_name = driver.find_element(By.ID, "firstHeading").text
    composer_name = composer_name.replace("Category:", "")
    ui.Select(driver.find_element(By.ID, "cathassecselp1")).select_by_value('R')
    try:
        next_button = driver.find_element(By.PARTIAL_LINK_TEXT, "next")
    except NoSuchElementException:
        next_button = None
    while True:
        all_links = driver.find_elements(By.TAG_NAME, 'a')
        i = 0
        while i < len(all_links):
            composition_number = int(driver.find_element(By.ID, "catnummsgp1").get_attribute("textContent"))
            if (all_links[i].text) == '📻':
                if (all_links[i + 1].text).startswith('next'):
                    composition_start_ind = i + 2
                elif (all_links[i + 1].text).startswith('previous'):
                    composition_start_ind = i + 2
                else:
                    composition_start_ind = i + 1
            i += 1
        for works in all_links[composition_start_ind:composition_start_ind + composition_number]:
            works_link = driver.find_element(By.LINK_TEXT, works.text)
            print(works_link.text)
            work_name = works_link.text
            if works_link.text in bad_link_list:
                continue
            works_link.send_keys(Keys.CONTROL + Keys.RETURN)
            works_link.send_keys(Keys.CONTROL + Keys.TAB)
            driver.switch_to.window(driver.window_handles[1])
            time.sleep(2)
            if composer_name.startswith('Bach') or composer_name.startswith('Vivaldi'):
                Period = 'Baroque'
            elif composer_name.startswith('Mozart') or composer_name.startswith('Beethoven'):
                Period = 'Classical'
            else:
                Period = 'Romantic'
            try:
                recording_link = driver.find_element(By.XPATH, "//span[contains(@title, 'Download this file')]/ancestor::a[@rel = 'nofollow']")
                print(recording_link.text, '\n')
                if (recording_link.text.endswith("(lossless)")):
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                    continue
            except NoSuchElementException:
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                continue
            if recording_link.text == '':
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
                #time.sleep(5)
                continue
            recording_link.send_keys(Keys.CONTROL + Keys.RETURN)
            driver.switch_to.window(driver.window_handles[2])
            #time.sleep(5)
            try:
                link = driver.find_element(By.LINK_TEXT, 'I accept this disclaimer, continue to download file')
                if link == None:
                    if len(driver.window_handles) == 3 and driver.current_window_handle == driver.window_handles[2]:
                        driver.close()
                        driver.switch_to.window(driver.window_handles[1])
                    # time.sleep(5)
                    if len(driver.window_handles) == 2 and driver.current_window_handle == driver.window_handles[1]:
                        driver.close()
                        driver.switch_to.window(driver.window_handles[0])
                        continue
                else:
                    link.click()
            except NoSuchElementException:
                pass
            try:
                driver.find_element(By.LINK_TEXT, 'I agree with the disclaimer above, continue my download').click()
                driver.switch_to.window(driver.window_handles[3])
                links = driver.find_elements(By.TAG_NAME, 'a')
                for link in links:
                    print(link.text)
                if len(driver.window_handles) == 4 and driver.current_window_handle == driver.window_handles[3]:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[2])
                if len(driver.window_handles) == 3 and driver.current_window_handle == driver.window_handles[2]:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[1])
                # time.sleep(5)
                if len(driver.window_handles) == 2 and driver.current_window_handle == driver.window_handles[1]:
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])
                # time.sleep(5)
                continue
            except NoSuchElementException:
                pass
            links = driver.find_elements(By.TAG_NAME, 'a')
            for link in links:
                print(link.text)

            try:
                file_path = Path(str(Music_ID) + '-' + composer_name + '.mp3')
                if not file_path.is_file():
                    audio_link = driver.find_element(By.XPATH, "//source[contains(@type, 'audio/mpeg')]")
                    audio_link = audio_link.get_attribute('src')
                    audio_file = requests.get(audio_link)
                    with open(str(Music_ID) + '-' + composer_name + '.mp3', 'wb') as file:
                        file.write(audio_file.content)
                data = [Music_ID, work_name, composer_name, Period]
                print(data)
                writer.writerow(data)
                Music_ID += 1
            except NoSuchElementException:
                pass
            if len(driver.window_handles) == 3 and driver.current_window_handle == driver.window_handles[2]:
                driver.close()
                driver.switch_to.window(driver.window_handles[1])
            #time.sleep(5)
            if len(driver.window_handles) == 2 and driver.current_window_handle == driver.window_handles[1]:
                driver.close()
                driver.switch_to.window(driver.window_handles[0])
            #time.sleep(5)

        if next_button == None:
            break
        next_button.click()
        try:
            next_button = driver.find_element(By.PARTIAL_LINK_TEXT, "next")
        except NoSuchElementException:
            next_button = None
label_csv.close()