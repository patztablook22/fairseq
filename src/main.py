import shellbot, discord
import sys, os
from configparser import ConfigParser

def main(argv):
    project = argv[1]
    config = ConfigParser()
    config.read(os.path.join(project, 'shellbot_config.ini'))
    token = open(os.path.join(project, 'shellbot_token.txt')).read().strip()
    admins = [int(l.strip()) for l in config['permissions']['admins'].splitlines() if l]
    shell = shellbot.Shellbot(admins=admins)

    # the outer shell loops this python script as long as it returns 69
    shell.run(token)
    if shell.restart: exit(69)

if __name__ == '__main__':
    main(sys.argv)

