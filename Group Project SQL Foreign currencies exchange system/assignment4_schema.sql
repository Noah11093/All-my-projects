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
drop trigger aa_availablefx_trg;
drop trigger aa_accountholding_trg;
drop trigger aa_transaction_availablefx_trg;

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
    constraint accountholding_id_ccy_pk PRIMARY KEY(accountid,ccy)
);

create table aa_accountholding_audit(
    seq number(8,0),
    accountid number(5,0),
    ccy varchar2(3),
    balance_before number(18,8),
    balance_after number(18,8),
    lastupdate date,
    constraint accountholding_audit_seq_pk PRIMARY KEY(seq)
);

create index holdingaudit_accid_date_i on aa_accountholding_audit (accountid, lastupdate);

CREATE SEQUENCE aa_accountholding_audit_seq
  START WITH 1
  INCREMENT BY 1
  MAXVALUE 99999999
  NOCYCLE
  NOCACHE;

/

create or replace trigger aa_accountholding_trg
after insert or update on aa_accountholding
for each row
declare

begin
    insert into aa_accountholding_audit 
    values
    (aa_accountholding_audit_seq.nextval, :new.accountid,
     :new.ccy, :old.balance, :new.balance, sysdate);
end;

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
    constraint transaction_txnid_pk PRIMARY KEY(transactionid)
);
-- Sequence for txnid, starts with 10001, max=99999999, not recycle
-- create or replace sequence xxx

CREATE SEQUENCE aa_transaction_seq
  START WITH 10001
  INCREMENT BY 1
  MAXVALUE 99999999
  NOCYCLE
  NOCACHE;

create index transaction_accountid_txndate_i on aa_transaction (accountid, transactiondate);

create table aa_availablefx(
    ccy varchar2(3),
    balance number(18,8),
    lastudpate date,
    constraint availablefx_ccy_pk PRIMARY KEY(ccy)
);

create table aa_availablefx_audit(
    seq number(8,0),
    ccy varchar2(3),
    balance_before number(18,8),
    balance_after number(18,8),
    lastudpate date,
    constraint availablefx_audit_seq_pk PRIMARY KEY(seq)
);

/

create or replace trigger aa_transaction_availablefx_trg
after insert on aa_transaction
for each row
declare

begin
    update aa_availablefx 
    set balance = balance + :new.amount1
    where ccy = :new.ccy1;
    
    update aa_availablefx 
    set balance = balance - :new.amount2
    where ccy = :new.ccy2;
end;

CREATE SEQUENCE aa_availablefx_audit_seq
  START WITH 1
  INCREMENT BY 1
  MAXVALUE 99999999
  NOCYCLE
  NOCACHE;

/

create or replace trigger aa_availablefx_trg
after insert or update on aa_availablefx
for each row
declare

begin
    insert into aa_availablefx_audit 
    values
    (aa_availablefx_audit_seq.nextval, :new.ccy, :old.balance, :new.balance, sysdate);
end;