



if __name__ == "__main__":
    modo = str(input())
    if modo == ("HSV" or "hsv"):
        pass
    elif modo == ("CIE" or "cie" or " CIE L*c*h*"):
        pass
    else:
        raise ValueError("Modo incorrecto, ingrese de nuevo el modo")
    