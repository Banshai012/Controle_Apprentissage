import multiprocessing
from drone import train

def start_instance(i):
    # Epsilon dégressif : la 1ère instance à 1.0, la dernière proche de 0.05
    eps = 0 
    print(f"Lancement de l'instance {i} avec Epsilon = {eps:.2f}")
    train(epsilon_start=max(eps, 0.01), duration_minutes=60)

if __name__ == '__main__':
    for _ in range(10):
        processes = []
        for i in range(5):
            p = multiprocessing.Process(target=start_instance, args=(i,))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()