import os
import glob
import xml.etree.ElementTree as ET

path = r'C:\Users\Administrator\Desktop\googlegarbage\1'
os.chdir(path)
pre='google_fouledplastic_'
def xml_to_txt(path):
    for xml_file in glob.glob(path + '\*.xml'):
        xml_pre_name = xml_file.split('_')[2]
        tree = ET.parse(xml_file)
        print(xml_file)
        root = tree.getroot()
        str1=root[1].text
        v1= pre+str1
        root[1].text = v1
        # tree.write(open(xml_pre_name))
        tree.write(xml_pre_name, encoding='utf-8', xml_declaration=False)


if __name__ == '__main__':
    xml_to_txt(path)
    
