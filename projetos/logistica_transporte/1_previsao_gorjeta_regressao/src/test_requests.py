import requests

# Dados de exemplo para prever (como uma lista de objetos JSON)
data = [
    {
        "segundos_da_viagem": 300,
        "milhas_da_viagem": 1.5,
        "area_comunitaria_do_embarque": 200,
        "area_comunitaria_do_desembarque": 300,
        "tarifa": 10.0,
        "cobrancas_adicionais": 2.5,
        "total_da_viagem": 12.5,
        "latitude_do_centroide_do_embarque": 40.7128,
        "longitude_do_centroide_do_embarque": -74.0060,
        "latitude_do_centroide_do_desembarque": 40.7128,
        "longitude_do_centroide_do_desembarque": -74.0060,
        "trato_do_censo_do_embarque": 1,
        "trato_do_censo_do_desembarque": 2,
        "hora_dia_inicio": 15,
        "dia_semana_inicio": 3,
        "viagens_agrupadas": "2023-11-25",
        "ano_inicio": 2023,
        "mes_inicio": 11,
        "dia_inicio": 25,
        "ano_final": 2023,
        "mes_final": 11,
        "dia_final": 25,
        "local_do_centroide_do_embarque": "Local A",
        "local_do_centroide_do_desembarque": "Local B"
    }
]

for _ in range(10):  # Enviar múltiplas requisições
    response = requests.post("http://127.0.0.1:5000/predict", json=data)
    print(response.json())
