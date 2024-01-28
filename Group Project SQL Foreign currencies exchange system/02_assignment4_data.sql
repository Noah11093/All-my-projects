truncate table aa_account;
truncate table aa_fxrate;
truncate table aa_availablefx;

INSERT INTO aa_account VALUES (aa_account_seq.NEXTVAL, 'Noah','Tsang','937 Progress Ave', 'Ontario','647-123-4567');
INSERT INTO aa_account VALUES (aa_account_seq.NEXTVAL, 'Thomas','Wong','23 Woodbine Road', 'Ontario','456-222-3439');
INSERT INTO aa_account VALUES (aa_account_seq.NEXTVAL, 'Kent','Wong','66 Eastwood Road', 'Alberta','567-221-3468');
INSERT INTO aa_account VALUES (aa_account_seq.NEXTVAL, 'Peter','Pan','104 Victoria Road', 'BC','468-876-8865');
INSERT INTO aa_account VALUES (aa_account_seq.NEXTVAL, 'Mickey','Mouse','88 Westwood Ave', 'PEI','123-778-3342');
INSERT INTO aa_account VALUES (aa_account_seq.NEXTVAL, 'Michelle','Jordan','123 Richmond Road', 'BC','089-122-9789');
INSERT INTO aa_account VALUES (aa_account_seq.NEXTVAL, 'Forest','Gump','1800 Sheppard Ave', 'Toronto','647-778-9909');
INSERT INTO aa_account VALUES (aa_account_seq.NEXTVAL, 'Ben','Stiller','979 Royal Road', 'PEI','998-114-5356');
INSERT INTO aa_account VALUES (aa_account_seq.NEXTVAL, 'Keira','Knightley','37 Museum Ave', 'YellowKnife','678-994-1213');
INSERT INTO aa_account VALUES (aa_account_seq.NEXTVAL, 'American','Captain','892 Apple Tree Road', 'Halifax','667-223-5354');

INSERT INTO aa_fxrate VALUES ('HKD','CAD',0.1746,1);
INSERT INTO aa_fxrate VALUES ('HKD','USD',0.1274,1);
INSERT INTO aa_fxrate VALUES ('HKD','JPY',16.6746,1);
INSERT INTO aa_fxrate VALUES ('HKD','GBP',0.1037,1);
INSERT INTO aa_fxrate VALUES ('HKD','EUR',0.1171,1);

INSERT INTO aa_fxrate VALUES ('CAD','HKD',5.7244,1);
INSERT INTO aa_fxrate VALUES ('CAD','USD',0.7292,1);
INSERT INTO aa_fxrate VALUES ('CAD','JPY',95.46,1);
INSERT INTO aa_fxrate VALUES ('CAD','GBP',0.5936,1);
INSERT INTO aa_fxrate VALUES ('CAD','EUR',0.6704,1);

INSERT INTO aa_fxrate VALUES ('USD','CAD',1.3709,1);
INSERT INTO aa_fxrate VALUES ('USD','HKD',7.8472,1);
INSERT INTO aa_fxrate VALUES ('USD','JPY',131.10,1);
INSERT INTO aa_fxrate VALUES ('USD','GBP',0.8139,1);
INSERT INTO aa_fxrate VALUES ('USD','EUR',0.9197,1);

INSERT INTO aa_fxrate VALUES ('JPY','CAD',1.0468,0.01);
INSERT INTO aa_fxrate VALUES ('JPY','HKD',5.9928,0.01);
INSERT INTO aa_fxrate VALUES ('JPY','USD',0.7625,0.01);
INSERT INTO aa_fxrate VALUES ('JPY','GBP',0.6214,0.01);
INSERT INTO aa_fxrate VALUES ('JPY','EUR',0.7018,0.01);

INSERT INTO aa_fxrate VALUES ('GBP','CAD',1.6846,1);
INSERT INTO aa_fxrate VALUES ('GBP','HKD',9.6460,1);
INSERT INTO aa_fxrate VALUES ('GBP','USD',1.2288,1);
INSERT INTO aa_fxrate VALUES ('GBP','JPY',160.93,1);
INSERT INTO aa_fxrate VALUES ('GBP','EUR',1.1294,1);

INSERT INTO aa_fxrate VALUES ('EUR','CAD',1.4915,1);
INSERT INTO aa_fxrate VALUES ('EUR','HKD',8.5373,1);
INSERT INTO aa_fxrate VALUES ('EUR','USD',1.0876,1);
INSERT INTO aa_fxrate VALUES ('EUR','JPY',142.48,1);
INSERT INTO aa_fxrate VALUES ('EUR','GBP',0.8854,1);


INSERT INTO aa_availablefx VALUES ('CAD',100000,SYSDATE);
INSERT INTO aa_availablefx VALUES ('HKD',100000,SYSDATE);
INSERT INTO aa_availablefx VALUES ('USD',100000,SYSDATE);
INSERT INTO aa_availablefx VALUES ('JPY',10000000,SYSDATE);
INSERT INTO aa_availablefx VALUES ('GBP',100000,SYSDATE);
INSERT INTO aa_availablefx VALUES ('EUR',100000,SYSDATE);

--insert into aa_accountholding values (103, 'CAD', 10000, sysdate);
--insert into aa_transaction values (aa_transaction_seq.nextval, 103, 'CAD', 'HKD', 5.7244, 1000, 5724.4, sysdate);
--insert into aa_transaction values (aa_transaction_seq.nextval, 103, 'HKD', 'CAD', .1746, 5724.4, 999.48024, sysdate);

commit;