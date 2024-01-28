drop table aa_account cascade constraints;
drop table aa_accountholding cascade constraints;
drop table aa_accountholding_audit cascade constraints;
drop table aa_fxrate cascade constraints;
drop table aa_transaction cascade constraints;
drop table aa_availablefx cascade constraints;
drop table aa_availablefx_audit cascade constraints;
drop sequence aa_transaction_seq;
drop sequence aa_account_seq;
drop sequence aa_availablefx_audit_seq;
drop sequence aa_accountholding_audit_seq;


create table aa_account (
    accountid number(5,0),
    firstname varchar2(50),
    lastname varchar2(50),
    address1 varchar2(100),
    address2 varchar2(100),
    phone varchar2(15),

    constraint account_id_pk PRIMARY KEY(accountid)
);
-- Sequence for accountid, starts with 101, max=9999, recycle
-- create or replace sequence xxx
CREATE SEQUENCE aa_account_seq
  START WITH 101
  INCREMENT BY 1
  MAXVALUE 9999
  CYCLE
  NOCACHE;


create table aa_accountholding(
    accountid number(5,0),
    ccy varchar2(3),
    balance number(18,8),
    lastupdate date,
    constraint accountholding_id_ccy_pk PRIMARY KEY(accountid,ccy),
    constraint account_accountid_fk FOREIGN KEY(accountid) REFERENCES aa_account(accountid)
);


create table aa_accountholding_audit(
    seq number(8,0),
    accountid number(5,0),
    ccy varchar2(3),
    balance_before number(18,8),
    balance_after number(18,8),
    lastupdate date,
    constraint accountholding_audit_seq_pk PRIMARY KEY(seq),
    constraint acchold_accountidccy_fk FOREIGN KEY(accountid,ccy) REFERENCES aa_accountholding (accountid,ccy)
);

create index holdingaudit_accid_date_i on aa_accountholding_audit (accountid, lastupdate);

CREATE SEQUENCE aa_accountholding_audit_seq
  START WITH 1
  INCREMENT BY 1
  MAXVALUE 99999999
  NOCYCLE
  NOCACHE;


create table aa_fxrate(
    ccy1 varchar2(3),
    ccy2 varchar2(3),
    rate number(18,8),
    multiplyby number(18,8),
    constraint fxrate_ccy1_ccy2_pk PRIMARY KEY(ccy1,ccy2)
);

create table aa_transaction(
    transactionid number(10,0),
    accountid number(5,0),
    ccy1 varchar2(3),
    ccy2 varchar2(3),
    rate number(18,8),
    amount1 number(18,8),
    amount2 number(18,8),
    transactiondate date,
    constraint transaction_txnid_pk PRIMARY KEY(transactionid),
    constraint acc_accountid_fk FOREIGN KEY(accountid) REFERENCES aa_account(accountid),
    constraint fxrate_ccy1ccy2_fk FOREIGN KEY(ccy1,ccy2) REFERENCES aa_fxrate (ccy1,ccy2)
);
-- Sequence for txnid, starts with 10001, max=99999999, not recycle
-- create or replace sequence xxx

CREATE SEQUENCE aa_transaction_seq
  START WITH 10001
  INCREMENT BY 1
  MAXVALUE 99999999
  NOCYCLE
  NOCACHE;

create index transaction_accid_txndate_i on aa_transaction (accountid, transactiondate);

create table aa_availablefx(
    ccy varchar2(3),
    balance number(18,8),
    lastupdate date,
    constraint availablefx_ccy_pk PRIMARY KEY(ccy)
);

create table aa_availablefx_audit(
    seq number(8,0),
    ccy varchar2(3),
    balance_before number(18,8),
    balance_after number(18,8),
    lastudpate date,
    constraint availablefx_audit_seq_pk PRIMARY KEY(seq),
    constraint availablefx_ccy_fk FOREIGN KEY(ccy) REFERENCES aa_availablefx (ccy)
);


CREATE SEQUENCE aa_availablefx_audit_seq
  START WITH 1
  INCREMENT BY 1
  MAXVALUE 99999999
  NOCYCLE
  NOCACHE;

