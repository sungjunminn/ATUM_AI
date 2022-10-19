def fix(item_name_list):
    for z in range(len(item_name_list)):
        if item_name_list[z] == "EFORLOOP":
            item_name_list[z] = "FORLOOP"
        if item_name_list[z] == "EFORLOOP-BODY":
            item_name_list[z] = "FORLOOP-BODY"
        if item_name_list[z] == "EFOR-MAINBLOCK":
            item_name_list[z] = "FOR-MAINBLOCK"
        if item_name_list[z] == "SETSUCESSCOUNT":
            item_name_list[z] = "SETSUCCESSCOUNT"
        if item_name_list[z] == "CONDITIONALLOGERROR":
            item_name_list[z] = "VALIDATIONCHECK"
        if item_name_list[z] == "VALIDATIONCHECK-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "MAPPUT-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "if":
            item_name_list[z] = "0"
        if item_name_list[z] == "ConditionalUDF-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "DBOBJECTSETS-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "FORLOOP-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "IFBLOCK-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "WHILELOOP-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "ELSEBLOCK-BODY":
            item_name_list[z] = "0"
        if item_name_list[z] == "ELSEIFBLOCK-BODY":
            item_name_list[z] = "0"
    while "0" in item_name_list:
        item_name_list.remove("0")
    return item_name_list
