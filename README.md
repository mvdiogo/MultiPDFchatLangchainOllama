## Python Conversational AI Software

This repository contains a Python software that utilizes Streamlit for a user interface and integrates various modules for conversational AI capabilities.

### Requirements
- Python 3.x
- Streamlit
- langchain
- langchain_community
- psycopg2
- dotenv

### Installation
1. Clone this repository to your local machine.
   ```bash
   git clone https://github.com/yourusername/your-repo.git
   ```
2. Navigate to the project directory.
   ```bash
   cd your-repo
   ```
3. Install the required Python packages.
   ```bash
   pip install -r requirements.txt
   ```
4. Install and configure [pgvector](https://github.com/pgvector/pgvector) on your computer.
   ```bash
	sudo apt-get install build-dep python-psycopg2
   ```


### Usage
1. Ensure you have set up your environment variables in a `.env` file. Required variables include `DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`, and `MODEL`.
2. Run the Streamlit application with the following command:
   ```bash
   streamlit run --theme.base dark app.py
   ```

### Components
- **Streamlit**: Used for building the user interface.
- **langchain**: Provides various functionalities for conversational AI, such as text splitting, memory management, and document loading.
- **langchain_community**: Offers additional features for conversational AI, including embeddings, LLMS, vector stores, and document loaders.
- **psycopg2**: PostgreSQL adapter for Python, required for interacting with PostgreSQL database.
- **dotenv**: Loads environment variables from a `.env` file.

### Environment Variables
- `DB_USER`: Username for the PostgreSQL database.
- `DB_PASSWORD`: Password for the PostgreSQL database.
- `DB_HOST`: Hostname of the PostgreSQL database server.
- `DB_PORT`: Port number of the PostgreSQL database.
- `DB_NAME`: Name of the PostgreSQL database.
- `MODEL`: Model information or configuration.

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgments
- Thanks to the developers of Streamlit, langchain, and langchain_community for their amazing contributions.
- Special thanks to the contributors who helped improve this software.