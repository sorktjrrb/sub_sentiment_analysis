
class Config :
    # 대문자로 적는 이유는 바뀌지 않는 상수를 뜻하기 위해
    # 앞으로는 접속 ID / PW 는 config로 관리 이 파일은 GIT hub에 올리면 안됨
    HOST = 'yh-db.cp46msk4u1po.ap-northeast-2.rds.amazonaws.com'
    DATABASE = 'jmdb_sub'
    DB_USER = 'jmdb_sub_user'
    DB_PASSWORD = '2340'

    # 실무에서 이 키값은 절대 노출되면 안됨
    SALT = 'sad;klfjsdlkf12@dkfj'

    # JWT 관련 변수 셋팅
    # 실무에서 이 키값은 절대 노출되면 안됨
    JWT_SECRET_KEY = 'yhschool,240522'
    # False로 설정하면 유효기간이 없고, True로 설정하면 유효기간이 생긴다.
    JWT_ACCESS_TOKEN_EXPIRES = False
    PROPAGATE_EXCEPTIONS = True