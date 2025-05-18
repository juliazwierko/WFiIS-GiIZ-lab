from Algorithms import generate_random_flow_network
from DrawGraph import *

if __name__ == '__main__':
    # Zadanie 1
    # Napisać program do tworzenia losowej sieci przepływowej między pojedynczym źródłem i pojedynczym ujściem według następującej procedury.
    # Na potrzeby programu wprowadzić warstwy, które idą od źródła do ujścia. Źródło znajduje się w zerowej warstwie a ujście w warstwie
    # N + 1. Liczba pośrednich warstw wynosi N i jest parametrem programu (N ≥ 2, a na potrzeby testowania N ≤ 4). Pośrednie warstwy
    # ponumerowane są od 1 do N. W każdej pośredniej warstwie rozmieścić losowo od dwóch do N wierzchołków. Połączyć wierzchołki kolejnych
    # warstw za pomocą łuków skierowanych od warstwy i do warstwy i + 1 (∀i = 0, . . . , N), tak aby z każdego wierzchołka leżącego w warstwie i
    # wychodził co najmniej jeden łuk i do każdego wierzchołka w warstwie i + 1 wchodził co najmniej jeden łuk. Do otrzymanego w ten sposób digrafu 
    # należy następnie dodać 2N łuków w sposób losowy. Łuki mają być losowane bez preferencji kierunku, tzn. nie muszą być skierowane zgodnie z warstwami. 
    # Należy jednak zwrócić uwagę, żeby nie dodać łuku już istniejącego i żeby nie dodać łuku wchodzącego do źródła albo wychodzącego z ujścia. 
    # Na tak otrzymanym digrafie przypisać każdemu łukowi liczbę naturalną z zakresu [1, 10] mającą interpretację przepustowości. Zakodować i narysować otrzymaną sieć.
    network = generate_random_flow_network(4, 0.5)
    Draw_Flow_Network(network, legend_title='Flow Network', filename = 'flow_network.png', output_dir='outputs/05')
    
    # Zadanie 2
