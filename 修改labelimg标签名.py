import glob
import xml.etree.ElementTree as ET
path = r'C:\Users\Administrator\Desktop\fortest'
for xml_file in glob.glob(path + '/*.xml'):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    for member in root.findall('object'):
        objectname = member.find('name').text
        if objectname == 'CigaretteButts': #原始标签名
            member.find('name').text = 'Cigarette'#修改后的标签名
            tree.write(xml_file,encoding='utf-8', xml_declaration=False)
