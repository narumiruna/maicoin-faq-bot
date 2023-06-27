from dotenv import load_dotenv

from maicoin_faq_bot.chain import MaiCoinFAQChain


def main():
    load_dotenv()

    chain = MaiCoinFAQChain.from_env()
    chain.run()


if __name__ == '__main__':
    main()
