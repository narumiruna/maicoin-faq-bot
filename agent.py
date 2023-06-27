from dotenv import load_dotenv

from maicoin_faq_bot.agent import MaiCoinFAQAgent


def main():
    load_dotenv()
    MaiCoinFAQAgent.from_env().run()


if __name__ == '__main__':
    main()
