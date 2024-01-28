/* 
 * Error code:
 *  -20001: Invalid account ID
 *  -20002: Invalid currency
 *  -20003: Other Error
 *  -20004: Invalid amount
 */
-- Function
-- Return the converted amount of CCY2
CREATE OR REPLACE
FUNCTION convertccy_fun
	(ccy1_name IN VARCHAR2, 
     ccy2_name IN VARCHAR2, 
     ccy1_amount IN NUMBER)
  RETURN NUMBER
  IS
	ccy2_amount aa_transaction.amount2%type;
	ccy_rate aa_fxrate.rate%type;
	ccy_multiplyby aa_fxrate.multiplyby%type;
BEGIN
    IF ccy1_name = ccy2_name THEN
        RETURN ccy1_amount;
    END IF;
    
	SELECT rate, multiplyby INTO ccy_rate, ccy_multiplyby
	FROM aa_fxrate 
	WHERE ccy1_name = ccy1 AND ccy2_name = ccy2;
	ccy2_amount := ccy1_amount * ccy_rate * ccy_multiplyby;
	RETURN ccy2_amount;

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RAISE_APPLICATION_ERROR(-20002, 'Invalid currency.');
END;
/

-- package
-- validate_fun(account ID)
-- validate_fun(currency name)
-- acc_bal_enq_prc(accout ID), list all currency holdings with base currency (CAD) value
-- movement_rpt_prc(account ID), provide a summary of the
create or replace package util_package is
    function validate_fun(accid in number) return number;
    function validate_fun(ccy in varchar2) return number;
    procedure acc_bal_enq_prc(accid in number);
    procedure movement_rpt_prc(accid in number);
    gv_base_cur varchar2(3) := 'CAD';
end util_package;
/
create or replace package body util_package is
    function validate_fun (accid in number)
    return number is
        lv_exist number(1,0);
    begin
        select nvl((select 1 from aa_account where accountid = accid),0) into lv_exist from dual;
        if lv_exist = 0 then
            dbms_output.put_line('Account ID: ' || accid || ' is not existed!');
        end if;
        return lv_exist;
    end validate_fun;
    
    function validate_fun (ccy in varchar2)
    return number is
        lv_exist number(1,0);
    begin
        select nvl((select distinct 1 from aa_fxrate where ccy1 = ccy or ccy2 = ccy),0) into lv_exist from dual;
        if lv_exist = 0 then
            dbms_output.put_line('Currency: ' || ccy || ' is not existed!');
        end if;
        return lv_exist;
    end validate_fun;
    
    procedure acc_bal_enq_prc(accid in number) is
        cursor cur_holding is
            select accountid, ccy, balance, lastupdate from aa_accountholding where accountid = accid;  
        rec_holding aa_accountholding%rowtype;
    begin
        if util_package.validate_fun(accid) = 1 then
            dbms_output.put_line('Account ID: ' || accid);
            open cur_holding;
            loop
                fetch cur_holding into rec_holding;
                    exit when cur_holding%notfound;
                    dbms_output.put_line(rec_holding.ccy || ': ' ||
                                         rec_holding.balance || 
                                         ' (' || util_package.gv_base_cur || ': ' ||
                                         convertccy_fun(rec_holding.ccy, util_package.gv_base_cur, rec_holding.balance) ||
                                         ')');
            end loop;
        else
            dbms_output.put_line('Invalid account id!');
        end if;
    end acc_bal_enq_prc;

    procedure movement_rpt_prc(accid in number) is
        cursor cur_holding is
            select seq, accountid, ccy, balance_before, balance_after, lastupdate 
            from aa_accountholding_audit 
            where accountid = accid
            order by ccy, lastupdate;  
        rec_holding aa_accountholding_audit%rowtype;
        lv_current_ccy varchar2(3) := '';
        lv_total_amt number(18,8) := 0;
        lv_txn_cnt number(10,0) := 0;
        lv_lastupdate date;
    begin
        if util_package.validate_fun(accid) = 1 then
            dbms_output.put_line('Account ID: ' || accid);
            open cur_holding;
            loop
                fetch cur_holding into rec_holding;
                    exit when cur_holding%notfound;
                    --dbms_output.put_line(lv_current_ccy || ' : ' || rec_holding.ccy);
                    if lv_current_ccy <> rec_holding.ccy then --and lv_current_ccy <> '' then
                        -- display corresponding currency summary
                        dbms_output.put_line(lv_current_ccy || ': ' || lv_total_amt);
                        dbms_output.put_line('Total transaction count for ' || lv_current_ccy || ': ' || lv_txn_cnt);
                        dbms_output.put_line('Last update date: ' || lv_lastupdate);
                        -- init the variables after displaying the records
                        lv_current_ccy := rec_holding.ccy;
                        lv_total_amt := 0;
                        lv_txn_cnt := 0;
                    end if;
                    lv_current_ccy := rec_holding.ccy;
                    lv_total_amt := lv_total_amt + rec_holding.balance_after - nvl(rec_holding.balance_before, 0);
                    lv_txn_cnt := lv_txn_cnt + 1;
                    lv_lastupdate := rec_holding.lastupdate;            
            end loop;
            -- display the last currency summary
            dbms_output.put_line(rec_holding.ccy || ': ' || lv_total_amt);
            dbms_output.put_line('Total transaction count for ' || lv_current_ccy || ': ' || lv_txn_cnt);
            dbms_output.put_line('Last update date: ' || lv_lastupdate);
        else
            dbms_output.put_line('Invalid account id!');
        end if;
    end movement_rpt_prc;
end util_package;
/


-- Function
-- Check the account contain sufficient funding
CREATE OR REPLACE
FUNCTION issufficient_fun
    (accid_num IN NUMBER, 
    ccy_id IN VARCHAR2, 
    amount_num IN NUMBER)
  RETURN NUMBER
  IS
    lv_exist number(1,0);
	acc_num aa_accountholding.accountid%type;
	ccy_name aa_accountholding.ccy%type;
	holding_bal aa_accountholding.balance%type;
BEGIN
    -- ensure all inputs are valid
    if util_package.validate_fun(accid_num) = 0 then
        RAISE_APPLICATION_ERROR(-20001, 'Invalid account ID.');
    end if;
    if util_package.validate_fun(ccy_id) = 0 then
        RAISE_APPLICATION_ERROR(-20002, 'Invalid currency.');
    end if;
    if amount_num <= 0 then
        RAISE_APPLICATION_ERROR(-20004, 'Invalid amount.');
    end if;
    -- check record exists
    select nvl((select 1 from aa_accountholding where accountid = accid_num and ccy = ccy_id),0) into lv_exist from dual;

    if lv_exist = 1 then
        SELECT accountid, ccy, balance INTO acc_num, ccy_name, holding_bal
        FROM aa_accountholding 
        WHERE accountid = accid_num AND ccy = ccy_id;
        
        if holding_bal >= amount_num then
            return 1;
        else
            return 0;
        end if;
    else
        return 0;
    end if;

END;
/
-- Function
-- Return the fx rate
CREATE OR REPLACE
FUNCTION getrate_fun
	(ccy1_name IN VARCHAR2,
     ccy2_name IN VARCHAR2)
  RETURN NUMBER
  IS
	ccy_rate aa_fxrate.rate%type;
BEGIN
    select rate into ccy_rate
    from aa_fxrate
    where ccy1 = ccy1_name and ccy2 = ccy2_name;    
	RETURN ccy_rate;
EXCEPTION
    WHEN NO_DATA_FOUND THEN
    	RAISE_APPLICATION_ERROR(-20002, 'Invalid currency.');
END;
/
-- Procedure
-- Insert the transaction information into table
CREATE OR REPLACE
PROCEDURE tradeccy_prc
	(acc_num IN NUMBER,
     ccy1_name IN VARCHAR2, 
     ccy2_name IN VARCHAR2, 
     ccy1_amount IN NUMBER)
AS
	acc_balance aa_accountholding.balance%type;
	ccy_rate aa_fxrate.rate%type;
	ccy_multiplyby aa_fxrate.multiplyby%type;
	ccy2_amount aa_transaction.amount2%type;
BEGIN
    IF issufficient_fun(acc_num, ccy1_name, ccy1_amount) = 1 THEN
    	INSERT INTO aa_transaction VALUES 
        (aa_transaction_seq.NEXTVAL, 
        acc_num, 
        ccy1_name, 
        ccy2_name, 
        getrate_fun(ccy1_name, ccy2_name), 
        ccy1_amount, 
        convertccy_fun(ccy1_name, ccy2_name, ccy1_amount), 
        SYSDATE);
		DBMS_OUTPUT.PUT_LINE('Transaction succeeded');
	ELSE
        DBMS_OUTPUT.PUT_LINE('Unsufficient funding');
	END IF;
EXCEPTION
    WHEN NO_DATA_FOUND THEN
    	RAISE_APPLICATION_ERROR(-20002, 'Invalid currency.');
	WHEN OTHERS THEN
        --RAISE_APPLICATION_ERROR(-20003, 'Other Error.');
        RAISE;
END;
/
-- Procedure
-- Update the balance and insert the record into the table after transaction
CREATE OR REPLACE PROCEDURE moveccy_prc (
    p_accountid IN aa_accountholding.accountid%TYPE,
    p_ccy IN aa_availablefx.ccy%TYPE,
    p_amount IN NUMBER
) AS
    lv_exist number(1,0);
    lv_amount NUMBER(18, 8);
    lv_iscredit number(1,0);
BEGIN
    -- ensure all inputs are valid
    if util_package.validate_fun(p_accountid) = 0 then
        RAISE_APPLICATION_ERROR(-20001, 'Invalid account ID.');
    end if;
    if util_package.validate_fun(p_ccy) = 0 then
        RAISE_APPLICATION_ERROR(-20002, 'Invalid currency.');
    end if;
    
    if p_amount > 0 then
        lv_amount := p_amount;
        lv_iscredit := 1;
    else
        lv_amount := p_amount * -1;
        lv_iscredit := 0;
    end if;
    
    -- Retrieve the current available balance for the given currency
    select nvl((select 1 from aa_accountholding where accountid = p_accountid and ccy = p_ccy),0) into lv_exist from dual;
    
    -- handle credit
    if lv_iscredit = 1 then
        if lv_exist = 1 then
            UPDATE aa_accountholding SET
            balance = balance + lv_amount,
            lastupdate = sysdate
            WHERE accountid = p_accountid AND ccy = p_ccy;
        else
            INSERT INTO aa_accountholding VALUES (p_accountid, p_ccy, p_amount, sysdate);
        end if;
    else
        -- handle debit
        if issufficient_fun(p_accountid, p_ccy, lv_amount) = 1 then
            UPDATE aa_accountholding SET
            balance = balance - lv_amount,
            lastupdate = sysdate
            WHERE accountid = p_accountid AND ccy = p_ccy;
        else
            DBMS_OUTPUT.PUT_LINE('insufficient fund!');
        end if;
    end if;
    
    DBMS_OUTPUT.PUT_LINE('moveccy_prc completed');
EXCEPTION
    WHEN NO_DATA_FOUND THEN
    	RAISE_APPLICATION_ERROR(-20002, 'Invalid currency.');
	WHEN OTHERS THEN
        --RAISE_APPLICATION_ERROR(-20003, 'Other Error.');
        RAISE;
END moveccy_prc;
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
/



create or replace trigger aa_transaction_achold_trg
after insert on aa_transaction
for each row
declare
    lv_exist number(1,0);
begin

    -- handling debit transaction
    update aa_accountholding set 
    balance = balance - :new.amount1,
    lastupdate = sysdate
    where accountid = :new.accountid;
    
    -- handling credit tranasction
    select nvl((select 1 from aa_accountholding where accountid = :new.accountid and ccy = :new.ccy2),0) into lv_exist from dual;

    if lv_exist = 1 then
        update aa_accountholding set
        balance = balance + :new.amount2,
        lastupdate = sysdate
        where accountid = :new.accountid and ccy = :new.ccy2;
    else
        insert into aa_accountholding values (:new.accountid, :new.ccy2, :new.amount2, sysdate);
    end if;
END;
/