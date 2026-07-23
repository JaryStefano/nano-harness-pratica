passo = 1

while True:
    print(f"\nPensando... (Passo {passo})")

    resposta = input("Digite a resposta da IA: ")

    if resposta == "fim":
        print("A IA terminou!")
        break

    print(f"A IA respondeu: {resposta}")

    passo += 1