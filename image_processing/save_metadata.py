import xml.etree.ElementTree as ET
import xml.dom.minidom

def save_xml_element(xml_element, saving_path, encoding="utf-8", xml_declaration=True, **kwargs):
    
    # Wrap xml_element in an ElementTree object
    tree = ET.ElementTree(xml_element)

    #  Save to disk as XML
    tree.write(saving_path, encoding=encoding, xml_declaration=xml_declaration, **kwargs)


def save_pretty_xml_element(xml_element, saving_path, indent="   ", encoding="utf-8"):
    # Convert xml_element to string
    xml_str = ET.tostring(xml_element, encoding=encoding)

    # Parse and pretty print
    parsed = xml.dom.minidom.parseString(xml_str)
    pretty_xml_as_string = parsed.toprettyxml(indent=indent)

    # Save to file
    with open(saving_path, "w", encoding=encoding) as f:
        f.write(pretty_xml_as_string)


def save_xml_string(xml_str, saving_path, indent="   ", encoding="utf-8"):
    
    # Parse and pretty print
    parsed = xml.dom.minidom.parseString(xml_str)
    pretty_xml_as_string = parsed.toprettyxml(indent=indent)

    # Save to file
    with open(saving_path, "w", encoding=encoding) as f:
        f.write(pretty_xml_as_string)