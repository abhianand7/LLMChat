from encryptoenv.EnvFile import EnvFile


def load_env_variables():
    env_file = EnvFile()
    env_file.create_environment_variables()


if __name__ == '__main__':
    load_env_variables()
